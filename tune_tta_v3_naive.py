"""
tune_tta_v3_naive.py - Random search for MergeSlide-TTA v3 CLASS-IL naive mode.

This tuner is intentionally single-phase. Naive inference does not use TCP
routing, so task-routing parameters are not searched. In addition, gamma_task is
forced to 0.0 in every trial so the adaptation objective does not optimize the
task-prompt margin loss while evaluating naive/global-head inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

import tune_tta_v3 as tcp_tuner


SEARCH_SPACE_NAIVE: dict[str, list] = {
    "eta_base":    [1e-5, 5e-5, 1e-4, 3e-4, 5e-4],
    "delta":       [0.01, 0.03, 0.05, 0.10, 0.20],
    "tau_c":       [0.05, 0.07, 0.10, 0.20, 0.30],
    "tau_ood":     [0.20, 0.30, 0.50, 0.70],
    "gamma_class": [0.5, 1.0, 2.0],
}

SHARED_PARSE_SPACE: dict[str, list] = {
    **SEARCH_SPACE_NAIVE,
    "gamma_task": [0.0],
}

FIXED_NAIVE_TTA: dict[str, float] = {
    # Naive does not use TCP routing. Disable task-margin adaptation explicitly.
    "gamma_task": 0.0,
}


def configure_shared_search_space() -> None:
    """Point imported helper functions at the naive-only search space."""
    tcp_tuner.SEARCH_SPACE = SHARED_PARSE_SPACE


def sample_config(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE_NAIVE.items()}


def load_default_tta_params(base_config_path: str) -> dict:
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    raw_tta = base_cfg.get("tta", {}) if isinstance(base_cfg, dict) else {}
    return {k: raw_tta.get(k) for k in SEARCH_SPACE_NAIVE.keys() if k in raw_tta}


def build_manifest(
    n_trials: int,
    seed: int,
    excluded_params: dict | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    manifest: list[dict] = []
    trial_id = 0
    excluded_count = 0
    while len(manifest) < n_trials:
        params = sample_config(rng)
        if excluded_params and all(
            params.get(k) == excluded_params.get(k)
            for k in SEARCH_SPACE_NAIVE.keys()
        ):
            excluded_count += 1
            continue
        manifest.append({"trial_id": trial_id, "params": params})
        trial_id += 1
    if excluded_count:
        print(f"[TTA Naive Tuning] excluded {excluded_count} default-config samples")
    return manifest


def make_trial_config_naive(
    base_config_path: str,
    params: dict,
    trial_dir: Path,
) -> str:
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    config.setdefault("tta", {})
    for key, value in params.items():
        config["tta"][key] = value
    for key, value in FIXED_NAIVE_TTA.items():
        config["tta"][key] = value

    trial_config_path = trial_dir / "trial_config.yaml"
    with open(trial_config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    return str(trial_config_path)


def write_manifest_csv(manifest: list[dict], manifest_csv_path: Path) -> None:
    with open(manifest_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", *SEARCH_SPACE_NAIVE.keys()])
        writer.writeheader()
        for item in manifest:
            writer.writerow({"trial_id": item["trial_id"], **item["params"]})


def run_one_trial_naive(
    trial_id: int,
    params: dict,
    setting: str,
    base_config: str,
    merge_dir: str,
    swag_dir: str,
    finetuned_dir: str,
    output_dir: Path,
    python_bin: str,
    project_root: str,
    num_folds: int,
    entrypoint_wrapper: str | None,
    no_reset_per_task: bool,
) -> dict:
    trial_dir = output_dir / f"trial_{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_config_path = make_trial_config_naive(base_config, params, trial_dir)
    result_csv = str(trial_dir / "results.csv")
    log_out = trial_dir / "stdout.log"
    log_err = trial_dir / "stderr.log"

    with open(trial_dir / "params.json", "w") as f:
        json.dump(
            {
                "trial_id": trial_id,
                "setting": setting,
                "mode": "classil_naive",
                **params,
                **FIXED_NAIVE_TTA,
            },
            f,
            indent=2,
        )

    cmd = [
        python_bin,
        "-u",
        entrypoint_wrapper if entrypoint_wrapper else "test_tta_v3.py",
    ]
    if entrypoint_wrapper:
        cmd += ["--entrypoint", "test_tta_v3.py"]
    cmd += [
        "--config",
        trial_config_path,
        "--save_dir",
        finetuned_dir,
        "--merge_model_path",
        merge_dir,
        "--swag_dir",
        swag_dir,
        "--mode",
        "classil_naive",
        "--result_csv",
        result_csv,
        "--fold_start",
        "0",
        "--fold_end",
        str(num_folds),
    ]
    if no_reset_per_task:
        cmd.append("--no_reset_per_task")

    print(f"\n{'=' * 60}")
    print(f"[Naive Trial {trial_id:04d}] Setting={setting} objective=bACC")
    for key, value in params.items():
        print(f"    {key:15s} = {value}")
    for key, value in FIXED_NAIVE_TTA.items():
        print(f"    {key:15s} = {value}  (fixed)")
    print(f"  config -> {trial_config_path}")
    print(f"{'=' * 60}")

    t0 = time.time()
    header_lines = [
        f"[INFO] start at {datetime.now()}",
        f"[INFO] command={' '.join(cmd)}",
        f"[INFO] trial_id={trial_id} setting={setting} mode=classil_naive",
        f"[INFO] params={json.dumps(params, sort_keys=True)}",
        f"[INFO] fixed={json.dumps(FIXED_NAIVE_TTA, sort_keys=True)}",
    ]
    try:
        stdout, stderr, returncode = tcp_tuner.run_command_streaming(
            cmd=cmd,
            cwd=project_root,
            stdout_path=Path(log_out),
            stderr_path=Path(log_err),
            header_lines=header_lines,
            timeout_sec=7200,
        )
    except Exception as exc:
        stdout = ""
        stderr = f"STREAMING_ERROR: {exc}"
        returncode = -1

    elapsed = time.time() - t0
    bacc = tcp_tuner.parse_bacc_from_output(stdout)
    task_baccs = tcp_tuner.parse_bacc_per_task(stdout)
    status = "ok" if returncode == 0 and bacc is not None else "failed"

    result = {
        "trial_id": trial_id,
        "setting": setting,
        "phase": "naive_class",
        "mode": "classil_naive",
        "status": status,
        "bacc_mean": bacc if bacc is not None else float("nan"),
        "routing_acc": float("nan"),
        "elapsed_s": elapsed,
        "returncode": returncode,
        **{f"task_{t}_bacc": task_baccs.get(t, float("nan")) for t in range(6)},
        **{f"task_{t}_routing_acc": float("nan") for t in range(6)},
        **params,
        **FIXED_NAIVE_TTA,
    }

    if bacc is not None:
        print(f"[Naive Trial {trial_id:04d}] -> bACC = {bacc:.4f}% ({elapsed / 60:.1f} min)")
    else:
        print(f"[Naive Trial {trial_id:04d}] -> FAILED (rc={returncode})")
        print(f"  stderr: {stderr[-400:]}")

    return result


def build_cached_trial_result_naive(
    trial_id: int,
    params: dict,
    setting: str,
    trial_dir: Path,
    result_csv: Path,
) -> dict:
    stdout_path = trial_dir / "stdout.log"
    stderr_path = trial_dir / "stderr.log"
    stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""

    bacc = tcp_tuner.parse_bacc_from_output(stdout)
    task_baccs = tcp_tuner.parse_bacc_per_task(stdout)
    status = "ok" if bacc is not None else "failed"
    elapsed = tcp_tuner._sum_elapsed_from_result_csv(result_csv)

    return {
        "trial_id": trial_id,
        "setting": setting,
        "phase": "naive_class",
        "mode": "classil_naive",
        "status": status,
        "bacc_mean": bacc if bacc is not None else float("nan"),
        "routing_acc": float("nan"),
        "elapsed_s": elapsed if elapsed is not None else None,
        "returncode": 0 if status == "ok" else 1,
        **{f"task_{t}_bacc": task_baccs.get(t, float("nan")) for t in range(6)},
        **{f"task_{t}_routing_acc": float("nan") for t in range(6)},
        **params,
        **FIXED_NAIVE_TTA,
        "cached_result_csv": str(result_csv),
        "cached_stdout": str(stdout_path),
        "cached_stderr_tail": stderr[-400:] if status != "ok" else "",
    }


def print_best_naive(results: list[dict], n: int = 5) -> None:
    valid = [
        row for row in results
        if not np.isnan(tcp_tuner._row_metric(row, "bacc_mean"))
    ]
    if not valid:
        print("  Chua co trial thanh cong.")
        return
    top = sorted(valid, key=lambda row: tcp_tuner._row_metric(row, "bacc_mean"), reverse=True)[:n]
    print(f"\n{'-' * 60}")
    print(f"TOP {min(n, len(top))} NAIVE TRIALS (objective=bACC):")
    for idx, row in enumerate(top):
        print(
            f"  #{idx + 1}  bACC={float(row.get('bacc_mean', 0)):.4f}%  "
            f"trial={int(row['trial_id']):04d}"
        )
        for key in SEARCH_SPACE_NAIVE:
            print(f"       {key:15s} = {row.get(key, 'N/A')}")
        print(f"       gamma_task      = {row.get('gamma_task', 'N/A')} (fixed)")
    print(f"{'-' * 60}\n")


def main() -> None:
    configure_shared_search_space()

    parser = argparse.ArgumentParser(
        description="Random search TTA v3 hyperparameter tuning for classil_naive"
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--setting", type=str, default="ind", choices=["ind", "ood"])
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--merge_dir", type=str, required=True)
    parser.add_argument("--swag_dir", type=str, required=True)
    parser.add_argument("--finetuned_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./logs/tune_tta_v3_naive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--python_bin", type=str, default=None)
    parser.add_argument("--entrypoint_wrapper", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--prepare_manifest", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_end", type=int, default=None)
    parser.add_argument(
        "--no_reset_per_task",
        action="store_true",
        help="Keep adapted model across tasks during each trial.",
    )
    args = parser.parse_args()

    if args.python_bin:
        python_bin = args.python_bin
    else:
        default_py = "/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
        python_bin = default_py if Path(default_py).exists() else sys.executable

    output_dir = Path(args.output_dir) / args.setting
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_start = args.trial_start
    trial_end = args.trial_end if args.trial_end is not None else args.n_trials
    manifest_path = Path(
        args.manifest_path
        if args.manifest_path is not None
        else output_dir / f"manifest_{args.setting}_naive_{args.n_trials}_{args.seed}.json"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = output_dir / f"summary_{timestamp}.csv"

    print("[TTA Naive Tuning] mode=classil_naive objective=bACC")
    print(f"[TTA Naive Tuning] Setting: {args.setting.upper()}")
    print(f"[TTA Naive Tuning] base_config: {args.base_config}")
    print(f"[TTA Naive Tuning] n_trials={args.n_trials} seed={args.seed} folds={args.num_folds}")
    print(f"[TTA Naive Tuning] trial_start={trial_start} trial_end={trial_end}")
    print(f"[TTA Naive Tuning] output_dir={output_dir}")
    print(f"[TTA Naive Tuning] manifest_path={manifest_path}")
    print(f"[TTA Naive Tuning] entrypoint_wrapper={args.entrypoint_wrapper or 'disabled'}")
    print(f"[TTA Naive Tuning] no_reset_per_task={args.no_reset_per_task}")
    print("[TTA Naive Tuning] fixed params:")
    for key, value in FIXED_NAIVE_TTA.items():
        print(f"  {key:15s}: {value}")
    print("\nSearch space:")
    for key, value in SEARCH_SPACE_NAIVE.items():
        print(f"  {key:15s}: {value}")

    if args.summarize_only:
        combined_rows = tcp_tuner.collect_worker_summaries(output_dir)
        combined_csv = output_dir / "summary_all.csv"
        tcp_tuner.write_combined_summary(combined_rows, combined_csv)
        tcp_tuner.write_best_trial_artifacts(combined_rows, output_dir, phase="class", top_k=5)
        print(f"[TTA Naive Tuning] combined summary -> {combined_csv}")
        print_best_naive(combined_rows, n=10)
        return

    if args.prepare_manifest:
        excluded_params = load_default_tta_params(args.base_config)
        manifest = build_manifest(args.n_trials, args.seed, excluded_params=excluded_params)
        tcp_tuner.save_manifest(manifest, manifest_path)
        write_manifest_csv(manifest, manifest_path.with_suffix(".csv"))
        print(f"[TTA Naive Tuning] manifest written -> {manifest_path}")
        print(f"[TTA Naive Tuning] manifest csv -> {manifest_path.with_suffix('.csv')}")
        return

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run with --prepare_manifest first."
        )

    manifest = tcp_tuner.load_manifest(manifest_path)
    if len(manifest) < args.n_trials:
        raise ValueError(f"Manifest has {len(manifest)} items but n_trials={args.n_trials}")

    all_results: list[dict] = []
    fieldnames = None

    for item in manifest[trial_start:trial_end]:
        trial_id = int(item["trial_id"])
        params = dict(item["params"])
        trial_dir = output_dir / f"trial_{trial_id:04d}"
        result_csv = trial_dir / "results.csv"

        if tcp_tuner.has_cached_trial_result(result_csv):
            print(f"[Naive Trial {trial_id:04d}] SKIP cached result -> {result_csv}")
            cached_row = build_cached_trial_result_naive(
                trial_id=trial_id,
                params=params,
                setting=args.setting,
                trial_dir=trial_dir,
                result_csv=result_csv,
            )
            all_results.append(cached_row)
            if fieldnames is None:
                fieldnames = list(cached_row.keys())
            write_header = not summary_csv.exists()
            with open(summary_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(cached_row)
            continue

        result = run_one_trial_naive(
            trial_id=trial_id,
            params=params,
            setting=args.setting,
            base_config=args.base_config,
            merge_dir=args.merge_dir,
            swag_dir=args.swag_dir,
            finetuned_dir=args.finetuned_dir,
            output_dir=output_dir,
            python_bin=python_bin,
            project_root=args.project_root,
            num_folds=args.num_folds,
            entrypoint_wrapper=args.entrypoint_wrapper,
            no_reset_per_task=args.no_reset_per_task,
        )
        all_results.append(result)

        if fieldnames is None:
            fieldnames = list(result.keys())
        write_header = not summary_csv.exists()
        with open(summary_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result)

        print_best_naive(all_results, n=5)

    print(f"\n{'=' * 60}")
    print(f"NAIVE TUNING COMPLETE - {args.n_trials} trials - {args.setting.upper()}")
    print(f"Summary -> {summary_csv}")
    print_best_naive(all_results, n=10)


if __name__ == "__main__":
    main()
