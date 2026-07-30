#!/usr/bin/env python3
"""Random search for TCP-only parameters with naive parameters frozen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from random_search_classil_tta import (
    has_cached_result,
    json_safe,
    params_to_cli,
    run_command_streaming,
)


SEARCH_SPACE: dict[str, list[Any]] = {
    "alpha": [0.1, 0.25, 0.5, 0.75, 1.0],
    "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
    "ema_alpha_prompt": [0.9, 0.99, 0.995, 0.999],
    "delta_margin": [0.05, 0.1, 0.15, 0.2, 0.3],
    "tp_anchor_beta": [0.0, 0.1, 0.3, 0.5, 0.7],
    "gamma_margin": [0.0, 0.01, 0.05, 0.1, 0.2],
    "tau_task": [0.5, 0.6, 0.7, 0.8, 0.9],
}

REQUIRED_NAIVE_PARAMS = {
    "M",
    "K_sub",
    "n_steps",
    "entropy_loss_weight",
    "tta_param_scope",
    "select_mode",
    "naive_use_task_entropy",
    "use_dapc",
    "lr",
    "l2_anchor_beta",
    "top_ratio",
    "entropy_threshold",
    "ema_alpha",
    "dapc_loss_weight",
    "dapc_tau_anchor",
    "dapc_beta",
}

NUM_FOLDS = 10
TASK_NAMES = ("BRCA", "RCC", "NSCLC", "ESCA", "TGCT", "CESC")
ESCA_TASK_ID = 3
CESC_TASK_ID = 5
OBJECTIVE_KEYS = (
    "esca_routing_mean",
    "cesc_routing_mean",
    "overall_routing_mean",
)


def load_naive_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params", payload)
    if not isinstance(params, dict):
        raise ValueError(f"Invalid naive parameter file: {path}")
    missing = sorted(REQUIRED_NAIVE_PARAMS - set(params))
    if missing:
        raise ValueError(
            f"Naive parameter file is missing required keys: {missing}"
        )
    overlap = sorted(set(params) & set(SEARCH_SPACE))
    if overlap:
        raise ValueError(
            "Naive parameter file unexpectedly contains TCP-only keys: "
            f"{overlap}"
        )
    return {key: params[key] for key in params}


def build_manifest(n_trials: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    max_combinations = math.prod(len(values) for values in SEARCH_SPACE.values())
    if n_trials > max_combinations:
        raise ValueError(
            f"Requested {n_trials} trials but only {max_combinations} unique "
            "TCP configurations exist."
        )
    keys = list(SEARCH_SPACE)
    manifest: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    while len(manifest) < n_trials:
        params = {key: rng.choice(SEARCH_SPACE[key]) for key in keys}
        signature = tuple(params[key] for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        manifest.append({"trial_id": len(manifest), "params": params})
    return manifest


def save_manifest(manifest: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["trial_id", *SEARCH_SPACE.keys()]
        )
        writer.writeheader()
        for item in manifest:
            writer.writerow({"trial_id": item["trial_id"], **item["params"]})


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid manifest format: {path}")
    return payload


def build_row(
    trial_id: int,
    setting: str,
    params: dict[str, Any],
    stdout: str,
    stderr: str,
    elapsed_s: float | None,
    returncode: int,
    trial_dir: Path,
    baseline: dict[str, float],
    baseline_tolerance: float,
) -> dict[str, Any]:
    routing = parse_routing_metrics(trial_dir / "results.csv")
    esca_mean = routing.get("esca_routing_mean", float("nan"))
    cesc_mean = routing.get("cesc_routing_mean", float("nan"))
    overall_mean = routing.get("overall_routing_mean", float("nan"))
    valid_metrics = all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (esca_mean, cesc_mean, overall_mean)
    )
    constraints_met = (
        valid_metrics
        and esca_mean >= baseline["esca_routing_mean"] - baseline_tolerance
        and cesc_mean >= baseline["cesc_routing_mean"] - baseline_tolerance
    )
    joint_score = harmonic_mean(
        (esca_mean, cesc_mean, overall_mean)
    )
    status = (
        "ok"
        if returncode == 0 and valid_metrics
        else "failed"
    )
    return {
        "trial_id": trial_id,
        "setting": setting,
        "mode": "tcp",
        "status": status,
        "objective": "constrained_routing_harmonic_mean",
        "objective_value": joint_score,
        "joint_score": joint_score,
        "constraints_met": constraints_met,
        "esca_routing_baseline": baseline["esca_routing_mean"],
        "cesc_routing_baseline": baseline["cesc_routing_mean"],
        "baseline_tolerance": baseline_tolerance,
        "elapsed_s": elapsed_s if elapsed_s is not None else float("nan"),
        "returncode": returncode,
        "stdout_log": str(trial_dir / "stdout.log"),
        "stderr_log": str(trial_dir / "stderr.log"),
        "result_csv": str(trial_dir / "results.csv"),
        "stderr_tail": stderr[-600:],
        **routing,
        **params,
    }


def harmonic_mean(values: tuple[float, ...]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return float("nan")
    return len(values) / sum(1.0 / value for value in values)


def parse_routing_metrics(result_csv: Path) -> dict[str, float]:
    if not result_csv.exists():
        return {}
    values: dict[tuple[int, int], float] = {}
    try:
        with result_csv.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                fold = int(row["fold"])
                task_id = int(row["task_id"])
                routing_acc = float(row["routing_acc"])
                if not 0 <= fold < NUM_FOLDS or not 0 <= task_id < len(TASK_NAMES):
                    return {}
                expected_name = TASK_NAMES[task_id]
                if row.get("task_name", expected_name) != expected_name:
                    return {}
                key = (fold, task_id)
                if key in values:
                    return {}
                values[key] = routing_acc
    except (KeyError, TypeError, ValueError):
        return {}

    expected = {
        (fold, task_id)
        for fold in range(NUM_FOLDS)
        for task_id in range(len(TASK_NAMES))
    }
    if set(values) != expected:
        return {}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        return {}

    result: dict[str, float] = {}
    task_means: dict[int, float] = {}
    for task_id, task_name in enumerate(TASK_NAMES):
        fold_values = [values[(fold, task_id)] for fold in range(NUM_FOLDS)]
        for fold, value in enumerate(fold_values):
            result[f"fold_{fold}_task_{task_id}_routing_acc"] = value
        task_means[task_id] = sum(fold_values) / NUM_FOLDS
        result[f"task_{task_id}_{task_name.lower()}_routing_mean"] = task_means[
            task_id
        ]
        result[f"task_{task_id}_{task_name.lower()}_routing_std"] = math.sqrt(
            sum((value - task_means[task_id]) ** 2 for value in fold_values)
            / NUM_FOLDS
        )

    result["esca_routing_mean"] = task_means[ESCA_TASK_ID]
    result["cesc_routing_mean"] = task_means[CESC_TASK_ID]
    result["overall_routing_mean"] = (
        sum(task_means.values()) / len(TASK_NAMES)
    )
    return result


def load_baseline_metrics(path: Path) -> dict[str, float]:
    metrics = parse_routing_metrics(path)
    if not metrics:
        raise ValueError(
            f"Baseline CSV must contain exactly {NUM_FOLDS} folds x "
            f"{len(TASK_NAMES)} tasks of valid routing results: {path}"
        )
    return {key: metrics[key] for key in OBJECTIVE_KEYS}


def run_trial(
    args: argparse.Namespace,
    trial_id: int,
    sampled_params: dict[str, Any],
    naive_params: dict[str, Any],
    output_dir: Path,
    baseline: dict[str, float],
) -> dict[str, Any]:
    trial_dir = output_dir / f"trial_{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    result_csv = trial_dir / "results.csv"
    params = {**naive_params, **sampled_params}
    (trial_dir / "params.json").write_text(
        json.dumps(
            json_safe(
                {
                    "trial_id": trial_id,
                    "setting": args.setting,
                    "mode": "tcp",
                    "naive_params_source": str(args.naive_params_file),
                    "fixed_naive_params": naive_params,
                    "sampled_tcp_params": sampled_params,
                    "params": params,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    if has_cached_result(result_csv):
        stdout_path = trial_dir / "stdout.log"
        stdout = (
            stdout_path.read_text(errors="replace")
            if stdout_path.exists()
            else ""
        )
        print(f"[Trial {trial_id:04d}] cached")
        return build_row(
            trial_id,
            args.setting,
            params,
            stdout,
            "",
            None,
            0,
            trial_dir,
            baseline,
            args.baseline_tolerance,
        )

    cmd = [
        args.python_bin,
        "-u",
        args.entrypoint_wrapper,
        "--entrypoint",
        "test_classIL_tta.py",
        "--config",
        args.base_config,
        "--save_dir",
        args.finetuned_dir,
        "--merge_model_path",
        args.merge_dir,
        "--mode",
        "tcp",
        "--result_csv",
        str(result_csv),
        "--efficiency_json",
        str(trial_dir / "efficiency.json"),
        "--tcp_tune_routing_all_folds",
        *params_to_cli(params),
    ]
    print(
        f"\n[Trial {trial_id:04d}] tcp={json.dumps(sampled_params, sort_keys=True)}"
    )
    started = time.time()
    try:
        stdout, stderr, returncode = run_command_streaming(
            cmd,
            args.project_root,
            trial_dir / "stdout.log",
            trial_dir / "stderr.log",
            args.timeout_sec,
        )
    except Exception as exc:
        stdout, stderr, returncode = "", str(exc), -1
    elapsed_s = time.time() - started
    row = build_row(
        trial_id,
        args.setting,
        params,
        stdout,
        stderr,
        elapsed_s,
        returncode,
        trial_dir,
        baseline,
        args.baseline_tolerance,
    )
    print(
        f"[Trial {trial_id:04d}] status={row['status']} "
        f"ESCA={row['esca_routing_mean'] * 100:.4f}% "
        f"CESC={row['cesc_routing_mean'] * 100:.4f}% "
        f"Overall={row['overall_routing_mean'] * 100:.4f}% "
        f"H={row['joint_score'] * 100:.4f}% "
        f"eligible={row['constraints_met']} "
        f"elapsed={elapsed_s / 60:.1f}m"
    )
    return row


def metric(row: dict[str, Any]) -> float:
    try:
        return float(row["joint_score"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def append_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(json_safe(row))


def collect_worker_rows(output_root: Path, setting: str) -> list[dict[str, Any]]:
    paths = list(output_root.glob(f"gpu*_w*/{setting}/tcp/summary_*.csv"))
    paths.extend(output_root.glob(f"{setting}/tcp/summary_*.csv"))
    rows_by_trial: dict[int, dict[str, Any]] = {}
    for path in sorted(paths):
        if path.name == "summary_all.csv":
            continue
        with path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    trial_id = int(float(row["trial_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
                rows_by_trial[trial_id] = row
    return [rows_by_trial[key] for key in sorted(rows_by_trial)]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def add_pareto_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows = [dict(row) for row in rows]
    remaining = set(range(len(ranked_rows)))
    pareto_rank = 1

    def objective_values(index: int) -> tuple[float, ...]:
        return tuple(float(ranked_rows[index][key]) for key in OBJECTIVE_KEYS)

    while remaining:
        front: list[int] = []
        for candidate in remaining:
            candidate_values = objective_values(candidate)
            dominated = False
            for other in remaining:
                if other == candidate:
                    continue
                other_values = objective_values(other)
                if all(
                    left >= right
                    for left, right in zip(other_values, candidate_values)
                ) and any(
                    left > right
                    for left, right in zip(other_values, candidate_values)
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        for index in front:
            ranked_rows[index]["pareto_rank"] = pareto_rank
            remaining.remove(index)
        pareto_rank += 1
    return ranked_rows


def write_best(
    output_dir: Path,
    rows: list[dict[str, Any]],
    top_k: int,
    naive_params: dict[str, Any],
) -> None:
    valid = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("objective") == "constrained_routing_harmonic_mean"
        and not math.isnan(metric(row))
        and all(
            math.isfinite(float(row[key]))
            for key in OBJECTIVE_KEYS
        )
    ]
    if not valid:
        raise ValueError("No successful TCP trial found.")
    pareto_rows = add_pareto_ranks(valid)
    write_rows(
        output_dir / "pareto_front.csv",
        [row for row in pareto_rows if int(row["pareto_rank"]) == 1],
    )

    eligible = [
        row
        for row in pareto_rows
        if as_bool(row.get("constraints_met"))
        and int(row["pareto_rank"]) == 1
    ]
    if not eligible:
        write_rows(output_dir / "constraint_failures.csv", pareto_rows)
        raise ValueError(
            "No trial satisfies both ESCA and CESC baseline constraints. "
            "See constraint_failures.csv; no best trial was selected."
        )

    ranked = sorted(
        eligible,
        key=lambda row: (
            metric(row),
            float(row["cesc_routing_mean"]),
            float(row["esca_routing_mean"]),
            float(row["overall_routing_mean"]),
            -int(float(row["trial_id"])),
        ),
        reverse=True,
    )
    best = dict(ranked[0])
    tcp_params = {key: best[key] for key in SEARCH_SPACE}
    full_params = {**naive_params, **tcp_params}
    best["fixed_naive_params"] = naive_params
    best["sampled_tcp_params"] = tcp_params
    best["params"] = full_params
    (output_dir / "best_trial.json").write_text(
        json.dumps(json_safe(best), indent=2), encoding="utf-8"
    )
    (output_dir / "best_tcp_params.json").write_text(
        json.dumps(json_safe(tcp_params), indent=2), encoding="utf-8"
    )
    (output_dir / "best_config.json").write_text(
        json.dumps(json_safe(full_params), indent=2), encoding="utf-8"
    )
    write_rows(output_dir / "top_k.csv", ranked[:top_k])
    print(
        f"[Random Search TCP] best trial={best['trial_id']} "
        f"ESCA={float(best['esca_routing_mean']) * 100:.4f}% "
        f"CESC={float(best['cesc_routing_mean']) * 100:.4f}% "
        f"Overall={float(best['overall_routing_mean']) * 100:.4f}% "
        f"H={float(best['joint_score']) * 100:.4f}% "
        f"ParetoRank={best['pareto_rank']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random search for TCP-only MergeSlide_TTA parameters."
    )
    parser.add_argument("--setting", choices=["ind", "ood"], default="ind")
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--merge_dir", required=True)
    parser.add_argument("--finetuned_dir", required=True)
    parser.add_argument("--naive_params_file", required=True)
    parser.add_argument(
        "--baseline_result_csv",
        default="",
        help="Routing baseline CSV with exactly 10 folds x 6 tasks.",
    )
    parser.add_argument(
        "--baseline_tolerance",
        type=float,
        default=0.0,
        help="Allowed absolute drop for ESCA/CESC routing versus baseline.",
    )
    parser.add_argument("--output_dir", default="./logs/tune_tcp")
    parser.add_argument(
        "--project_root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--entrypoint_wrapper",
        default="tools/run_classil_with_pt_features.py",
    )
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--prepare_manifest", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_end", type=int)
    parser.add_argument("--timeout_sec", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    args.timeout_sec = args.timeout_sec or None
    if args.baseline_tolerance < 0:
        parser.error("--baseline_tolerance must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    naive_params = load_naive_params(Path(args.naive_params_file))
    output_root = Path(args.output_dir)
    output_dir = output_root / args.setting / "tcp"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else output_dir
        / f"manifest_{args.setting}_tcp_{args.n_trials}_{args.seed}.json"
    )

    print(
        f"[Random Search TCP] setting={args.setting} trials={args.n_trials}"
    )
    print(f"[Random Search TCP] fixed_naive={naive_params}")
    print(f"[Random Search TCP] space={SEARCH_SPACE}")

    if args.prepare_manifest:
        save_manifest(build_manifest(args.n_trials, args.seed), manifest_path)
        print(f"[Random Search TCP] manifest={manifest_path}")
        return

    if not args.baseline_result_csv:
        raise ValueError(
            "--baseline_result_csv is required for TCP trial execution "
            "and summarization."
        )
    baseline_path = Path(args.baseline_result_csv)
    baseline = load_baseline_metrics(baseline_path)
    print(f"[Random Search TCP] baseline_csv={baseline_path}")
    print(f"[Random Search TCP] baseline={baseline}")
    print(
        f"[Random Search TCP] ESCA/CESC tolerance="
        f"{args.baseline_tolerance}"
    )

    if args.summarize_only:
        rows = collect_worker_rows(output_root, args.setting)
        if not rows:
            raise FileNotFoundError(
                f"No TCP worker summaries found under {output_root}"
            )
        write_rows(output_dir / "summary_all.csv", rows)
        write_best(output_dir, rows, args.top_k, naive_params)
        return

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Prepare it first."
        )
    manifest = load_manifest(manifest_path)
    trial_end = args.trial_end if args.trial_end is not None else args.n_trials
    if not 0 <= args.trial_start <= trial_end <= len(manifest):
        raise ValueError(
            f"Invalid trial range [{args.trial_start}, {trial_end}) for "
            f"manifest with {len(manifest)} trials."
        )

    summary_path = output_dir / (
        f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
    )
    rows: list[dict[str, Any]] = []
    for item in manifest[args.trial_start:trial_end]:
        row = run_trial(
            args,
            int(item["trial_id"]),
            dict(item["params"]),
            naive_params,
            output_dir,
            baseline,
        )
        rows.append(row)
        append_row(summary_path, row)
    write_rows(output_dir / "summary_latest.csv", rows)


if __name__ == "__main__":
    main()
