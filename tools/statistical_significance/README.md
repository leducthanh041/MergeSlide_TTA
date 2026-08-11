# Fold-wise statistical significance

This tool compares CAST-Slide TCP against IND baselines using matched folds.

Primary analysis:

- one-sided exact paired sign-flip test;
- Holm correction independently for `bacc`, `mean_acc`, and `macro_f1`;
- paired bootstrap confidence interval;
- Cohen's `d_z` effect size;
- Wilcoxon signed-rank and paired t-test as secondary diagnostics when SciPy is available.

Run from the repository root:

```bash
mkdir -p logs/statistical_significance/ind
python tools/statistical_significance/compute_ind_pvalues.py \
  --output-dir logs/statistical_significance/ind \
  2>&1 | tee logs/statistical_significance/ind/terminal.log
```

The script writes:

- `run.log`: detailed execution log;
- `source_files.csv`: exact input provenance;
- `coverage_report.csv`: available fold count for every metric and method;
- `ind_fold_metrics.csv`: normalized fold-level data;
- `ind_pvalues_all.csv`: all primary and secondary statistics;
- `ind_significance_table.csv`: compact paper-oriented table;
- `skipped_comparisons.csv`: comparisons rejected because fold-level data are missing;
- `ind_statistical_report.md`: readable report.

Do not use `--allow-partial` for paper results. Missing per-fold values are deliberately
not reconstructed from aggregate mean and standard deviation.

`fold_macro_f1` is available in CAST-Slide's result CSV after rerunning the current
`test_classIL_tta.py`. Older CSV files do not contain enough information to recover
fold-wise macroF1 and will be reported as incomplete.

Before reporting p-values, verify that every source uses the same definition. In
particular, a task-averaged metric is not interchangeable with a metric recomputed
from pooled global predictions.

IND reverse uses a separate manifest and treats lower FGT as better:

```bash
mkdir -p logs/statistical_significance/ind_reverse
python tools/statistical_significance/compute_ind_reverse_pvalues.py \
  --output-dir logs/statistical_significance/ind_reverse \
  2>&1 | tee logs/statistical_significance/ind_reverse/terminal.log
```

Holm correction is applied independently to `bacc`, `fgt`, `bwt`, and `macro_f1`.

OOD bACC only:

```bash
mkdir -p logs/statistical_significance/ood_bacc
python tools/statistical_significance/compute_ood_bacc_pvalues.py \
  --output-dir logs/statistical_significance/ood_bacc \
  2>&1 | tee logs/statistical_significance/ood_bacc/terminal.log
```
