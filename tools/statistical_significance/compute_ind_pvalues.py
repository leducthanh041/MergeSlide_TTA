#!/usr/bin/env python3
import argparse
import itertools
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


METRICS = ("bacc", "mean_acc", "macro_f1")
SUMMARY_RE = re.compile(
    r"\[SUMMARY\]\s+fold=(?P<fold>\d+).*?"
    r"bacc=(?P<bacc>[\d.]+)%.*?macro_f1=(?P<macro_f1>[\d.]+)%"
)
FOLD_LINE_RE = re.compile(
    r"\[Fold\s+(?P<fold>\d+)\]\s+Acc=(?P<acc>[\d.]+)%\s+BAcc=(?P<bacc>[\d.]+)%"
)
CLASS_IL_FOLD_RE = re.compile(
    r"\[Fold\s+(?P<fold>\d+)\]\s+Class-IL\s+ACC=(?P<acc>[\d.]+)%\s+"
    r"BAcc=(?P<bacc>[\d.]+)%"
)
MACC_RE = re.compile(
    r"\[Fold\s+(?P<fold>\d+)\]\s+mACC=(?P<mean_acc>[\d.]+)%"
)
RESULT_FOLD_RE = re.compile(r"(?:\[Fold\s+|\bfold=)(?P<fold>\d+)", re.IGNORECASE)
RESULT_BACC_RE = re.compile(r"\bbacc=(?P<value>[\d.]+)(?P<percent>%?)", re.IGNORECASE)
RESULT_MACRO_F1_RE = re.compile(
    r"\bmacro_?f1=(?P<value>[\d.]+)(?P<percent>%?)", re.IGNORECASE
)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fold-wise paired significance tests for CAST-Slide on IND."
    )
    parser.add_argument("--manifest", type=Path, default=here / "ind_sources.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Test methods with fewer than expected folds. Not recommended for paper results.",
    )
    return parser.parse_args()


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", mode="w"),
        ],
    )


def resolve_path(path_value, repo_root):
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else repo_root / path


def normalize_metric(values):
    values = pd.to_numeric(values, errors="coerce").astype(float)
    finite = values[np.isfinite(values)]
    if not finite.empty and finite.abs().max() > 1.5:
        values = values / 100.0
    return values


def read_csv_source(source, path):
    frame = pd.read_csv(path)
    fold_column = source.get("fold_column", "fold")
    if fold_column not in frame.columns:
        raise ValueError(f"missing fold column '{fold_column}'")
    output = pd.DataFrame({"fold": pd.to_numeric(frame[fold_column], errors="coerce")})
    for metric, column in source.get("metric_columns", {}).items():
        if column not in frame.columns:
            logging.warning("%s: missing optional metric column '%s' for %s", path, column, metric)
            continue
        output[metric] = normalize_metric(frame[column])
    if len(output.columns) == 1:
        raise ValueError("none of the configured metric columns were found")
    return output.dropna(subset=["fold"]).assign(fold=lambda x: x.fold.astype(int))


def read_task_csv_mean(source, path):
    frame = read_csv_source(source, path)
    metric_columns = [metric for metric in METRICS if metric in frame.columns]
    return frame.groupby("fold", as_index=False)[metric_columns].mean()


def read_log_source(path, pattern):
    rows = []
    for match in pattern.finditer(path.read_text(errors="replace")):
        row = {"fold": int(match.group("fold"))}
        if "bacc" in match.groupdict():
            row["bacc"] = float(match.group("bacc")) / 100.0
        if "macro_f1" in match.groupdict():
            row["macro_f1"] = float(match.group("macro_f1")) / 100.0
        if "mean_acc" in match.groupdict():
            row["mean_acc"] = float(match.group("mean_acc")) / 100.0
        rows.append(row)
    if not rows:
        raise ValueError("no fold-level metric lines matched")
    return pd.DataFrame(rows).drop_duplicates("fold", keep="last")


def parse_result_value(match):
    value = float(match.group("value"))
    return value / 100.0 if match.group("percent") or abs(value) > 1.5 else value


def read_result_metric_log(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        fold_match = RESULT_FOLD_RE.search(line)
        bacc_match = RESULT_BACC_RE.search(line)
        macro_f1_match = RESULT_MACRO_F1_RE.search(line)
        if fold_match is None or bacc_match is None:
            continue
        row = {
            "fold": int(fold_match.group("fold")),
            "bacc": parse_result_value(bacc_match),
        }
        if macro_f1_match is not None:
            row["macro_f1"] = parse_result_value(macro_f1_match)
        rows.append(row)
    if not rows:
        raise ValueError("no fold-level result metrics matched")
    return pd.DataFrame(rows).drop_duplicates("fold", keep="last")


def read_source(source, repo_root):
    path = resolve_path(source["path"], repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    source_type = source["type"]
    if source_type == "csv":
        frame = read_csv_source(source, path)
    elif source_type == "task_csv_mean":
        frame = read_task_csv_mean(source, path)
    elif source_type == "summary_log":
        frame = read_log_source(path, SUMMARY_RE)
    elif source_type == "fold_line_log":
        frame = read_log_source(path, FOLD_LINE_RE)
    elif source_type == "class_il_fold_log":
        frame = read_log_source(path, CLASS_IL_FOLD_RE)
    elif source_type == "macc_log":
        frame = read_log_source(path, MACC_RE)
    elif source_type == "result_metric_log":
        frame = read_result_metric_log(path)
    else:
        raise ValueError(f"unsupported source type: {source_type}")
    return frame, path


def merge_sources(method, repo_root):
    merged = None
    used_paths = []
    errors = []
    for source in method.get("sources", []):
        try:
            frame, path = read_source(source, repo_root)
            used_paths.append(str(path))
            merged = frame if merged is None else merged.merge(
                frame, on="fold", how="outer", validate="one_to_one"
            )
        except Exception as exc:
            errors.append(f"{source.get('path')}: {exc}")
    if merged is None:
        merged = pd.DataFrame(columns=["fold", *METRICS])
    for metric in METRICS:
        if metric not in merged:
            merged[metric] = np.nan
    merged.insert(0, "family", method["family"])
    merged.insert(0, "method", method["name"])
    return merged[["method", "family", "fold", *METRICS]], used_paths, errors


def exact_sign_flip_greater(differences):
    differences = np.asarray(differences, dtype=float)
    observed = differences.mean()
    count = 0
    total = 2 ** len(differences)
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = np.mean(differences * np.asarray(signs))
        count += statistic >= observed - tolerance
    return count / total


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_ci(differences, samples, seed):
    rng = np.random.default_rng(seed)
    differences = np.asarray(differences, dtype=float)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def significance_stars(p_value):
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


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
    diff = ref - base
    std_diff = diff.std(ddof=1)
    dz = diff.mean() / std_diff if std_diff > 0 else math.copysign(math.inf, diff.mean())
    ci_low, ci_high = bootstrap_ci(diff, samples, seed)
    p_wilcoxon = np.nan
    p_ttest = np.nan
    if stats is not None:
        try:
            p_wilcoxon = stats.wilcoxon(
                diff, alternative="greater", method="auto"
            ).pvalue
        except ValueError:
            pass
        p_ttest = stats.ttest_rel(ref, base, alternative="greater").pvalue

    return {
        "metric": metric,
        "n_folds": len(paired),
        "folds": ",".join(map(str, paired.fold.tolist())),
        "reference_mean": ref.mean(),
        "reference_std": ref.std(ddof=1),
        "baseline_mean": base.mean(),
        "baseline_std": base.std(ddof=1),
        "mean_difference": diff.mean(),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": dz,
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
        "p_signflip_raw": exact_sign_flip_greater(diff),
        "p_wilcoxon_raw": p_wilcoxon,
        "p_paired_t_raw": p_ttest,
    }, None


def write_report(output_dir, manifest, normalized, results, coverage):
    reference_name = manifest["reference_method"]
    lines = [
        "# IND fold-wise statistical significance report",
        "",
        f"Reference method: **{reference_name}**",
        "",
        "Primary test: one-sided exact paired sign-flip test over matched folds. ",
        "Holm correction is applied separately within each metric family. ",
        "Positive differences mean the reference method is better.",
        "",
        "## Coverage",
        "",
        "| Method | bACC folds | meanACC folds | macroF1 folds | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in coverage.to_dict("records"):
        lines.append(
            f"| {row['method']} | {row['bacc_folds']} | {row['mean_acc_folds']} | "
            f"{row['macro_f1_folds']} | {row['status']} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    if results.empty:
        lines.append("No complete paired comparisons were available.")
    else:
        lines.extend([
            "| Metric | Baseline | CAST-Slide | Baseline | Delta | 95% CI | p raw | p Holm | Sig. |",
            "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        for row in results.sort_values(["metric", "baseline"]).to_dict("records"):
            lines.append(
                f"| {row['metric']} | {row['baseline']} | "
                f"{100*row['reference_mean']:.3f} | {100*row['baseline_mean']:.3f} | "
                f"{100*row['mean_difference']:+.3f} | "
                f"[{100*row['ci95_low']:+.3f}, {100*row['ci95_high']:+.3f}] | "
                f"{row['p_signflip_raw']:.6g} | {row['p_holm']:.6g} | {row['stars']} |"
            )
    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- p-values are not probabilities that the null hypothesis is true.",
        "- A star is assigned only from the Holm-adjusted primary p-value.",
        "- Missing fold-level metrics are not reconstructed from aggregate mean and standard deviation.",
        "- Confirm that every method uses the same folds and identical metric definitions before paper submission.",
        "- Some TTA-guided logs report task-averaged bACC/macroF1, whereas other evaluators may report pooled global-class metrics. These must not be mixed without protocol verification.",
        "- With 10 folds, the smallest attainable exact sign-flip p-value is 1/1024.",
    ])
    (output_dir / "ind_statistical_report.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    repo_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text())
    expected_folds = int(manifest.get("expected_folds", 10))

    all_frames = []
    source_records = []
    method_errors = {}
    for method in manifest["methods"]:
        frame, used_paths, errors = merge_sources(method, repo_root)
        all_frames.append(frame)
        method_errors[method["name"]] = errors
        for path in used_paths:
            source_records.append({
                "method": method["name"], "family": method["family"], "path": path
            })
        logging.info(
            "%s: bACC=%d, meanACC=%d, macroF1=%d folds",
            method["name"],
            frame.bacc.notna().sum(),
            frame.mean_acc.notna().sum(),
            frame.macro_f1.notna().sum(),
        )
        for error in errors:
            logging.warning("%s source skipped: %s", method["name"], error)

    normalized = pd.concat(all_frames, ignore_index=True).sort_values(["method", "fold"])
    normalized.to_csv(args.output_dir / "ind_fold_metrics.csv", index=False)
    pd.DataFrame(source_records).to_csv(args.output_dir / "source_files.csv", index=False)

    coverage_rows = []
    for method in manifest["methods"]:
        frame = normalized[normalized.method == method["name"]]
        counts = {metric: int(frame[metric].notna().sum()) for metric in METRICS}
        status_parts = []
        for metric, count in counts.items():
            if count != expected_folds:
                status_parts.append(f"{metric}: {count}/{expected_folds}")
        if method_errors[method["name"]]:
            status_parts.append("source error")
        coverage_rows.append({
            "method": method["name"],
            "family": method["family"],
            "bacc_folds": counts["bacc"],
            "mean_acc_folds": counts["mean_acc"],
            "macro_f1_folds": counts["macro_f1"],
            "status": "; ".join(status_parts) or "complete",
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output_dir / "coverage_report.csv", index=False)

    reference_name = manifest["reference_method"]
    reference = normalized[normalized.method == reference_name]
    if reference.empty:
        raise RuntimeError(f"reference method not found: {reference_name}")

    result_rows = []
    skipped_rows = []
    for method in manifest["methods"]:
        if method["name"] == reference_name:
            continue
        baseline = normalized[normalized.method == method["name"]]
        for metric_index, metric in enumerate(METRICS):
            result, reason = compare_metric(
                reference,
                baseline,
                metric,
                expected_folds,
                args.allow_partial,
                args.bootstrap_samples,
                args.seed + metric_index,
            )
            if result is None:
                skipped_rows.append({
                    "baseline": method["name"], "metric": metric, "reason": reason
                })
                continue
            result.update({"baseline": method["name"], "family": method["family"]})
            result_rows.append(result)

    results = pd.DataFrame(result_rows)
    if not results.empty:
        adjusted_parts = []
        for _, part in results.groupby("metric", sort=False):
            part = part.copy()
            part["p_holm"] = holm_adjust(part.p_signflip_raw.to_numpy())
            part["stars"] = part.p_holm.map(significance_stars)
            adjusted_parts.append(part)
        results = pd.concat(adjusted_parts, ignore_index=True)
        results.to_csv(args.output_dir / "ind_pvalues_all.csv", index=False)
        results[[
            "metric", "baseline", "family", "n_folds", "reference_mean",
            "baseline_mean", "mean_difference", "ci95_low", "ci95_high",
            "cohen_dz", "wins", "ties", "losses", "p_signflip_raw",
            "p_holm", "stars"
        ]].to_csv(args.output_dir / "ind_significance_table.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "ind_pvalues_all.csv", index=False)
        pd.DataFrame().to_csv(args.output_dir / "ind_significance_table.csv", index=False)
    skipped = pd.DataFrame(skipped_rows)
    skipped.to_csv(args.output_dir / "skipped_comparisons.csv", index=False)

    write_report(args.output_dir, manifest, normalized, results, coverage)
    logging.info("Wrote results to %s", args.output_dir.resolve())
    logging.info("Complete comparisons: %d; skipped: %d", len(results), len(skipped))
    if stats is None:
        logging.warning("SciPy unavailable: Wilcoxon and paired t-test were not computed")


if __name__ == "__main__":
    main()
