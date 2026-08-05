#!/usr/bin/env python3
"""Random search for MergeSlide_TTA naive Class-IL using mean bACC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SEARCH_SPACE: dict[str, list[Any]] = {
    "lr": [2e-5, 5e-5, 1e-4, 2e-4],
    "l2_anchor_beta": [0.1, 0.3, 0.5, 1.0, 2.0],
    "top_ratio": [0.25, 0.5, 0.75, 1.0],
    "entropy_threshold": [0.2, 0.3, 0.4, 0.5, 0.7],
    "ema_alpha": [0.9, 0.99, 0.995, 0.999],
    "dapc_loss_weight": [0.25, 0.5, 1.0, 2.0],
    "dapc_tau_anchor": [0.5, 0.7, 0.8, 0.9, 0.92],
    "dapc_beta": [1.0, 1.2, 1.5, 2.0],
}

FIXED_PARAMS: dict[str, Any] = {
    "M": 8,
    "K_sub": 300,
    "n_steps": 3,
    "entropy_loss_weight": 1.0,
    "tta_param_scope": "ln_only",
    "select_mode": "intersection",
    "naive_use_task_entropy": True,
    "use_dapc": True,
}

BOOL_FLAGS = {
    "naive_use_task_entropy": "--naive_use_task_entropy",
    "use_dapc": "--use_dapc",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def build_manifest(n_trials: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    max_combinations = math.prod(len(values) for values in SEARCH_SPACE.values())
    if n_trials > max_combinations:
        raise ValueError(
            f"Requested {n_trials} trials but the search space contains "
            f"only {max_combinations} unique configurations."
        )

    manifest: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    keys = list(SEARCH_SPACE)
    while len(manifest) < n_trials:
        params = {
            key: rng.choice(SEARCH_SPACE[key])
            for key in keys
        }
        signature = tuple(params[key] for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        manifest.append(
            {"trial_id": len(manifest), "params": params}
        )
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
            writer.writerow(
                {"trial_id": item["trial_id"], **item["params"]}
            )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError(f"Invalid manifest format: {path}")
    return manifest


def parse_percent_line(stdout: str, marker: str) -> float | None:
    for line in stdout.splitlines():
        if marker not in line:
            continue
        try:
            return float(line.split(marker, 1)[1].split("%", 1)[0].strip())
        except (IndexError, ValueError):
            continue
    return None


def parse_task_bacc(result_csv: Path) -> dict[int, float]:
    if not result_csv.exists():
        return {}
    grouped: dict[int, list[float]] = {}
    try:
        with result_csv.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                task_id = int(row["task_id"])
                grouped.setdefault(task_id, []).append(float(row["bacc"]))
    except (KeyError, TypeError, ValueError):
        return {}
    return {
        task_id: sum(values) / len(values)
        for task_id, values in grouped.items()
        if values
    }


def parse_task_acc(stdout: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Task "):
            continue
        try:
            left, right = stripped.split(":", 1)
            task_id = int(left.replace("Task ", "").strip())
            values[task_id] = float(right.split("%", 1)[0].strip())
        except (IndexError, ValueError):
            continue
    return values


def has_cached_result(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", newline="") as handle:
            return any(True for _ in csv.DictReader(handle))
    except Exception:
        return False


def params_to_cli(params: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in params.items():
        if key in BOOL_FLAGS:
            if bool(value):
                cli.append(BOOL_FLAGS[key])
            continue
        cli.extend([f"--{key}", str(value)])
    return cli


def run_command_streaming(
    cmd: list[str],
    cwd: str,
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: int | None,
) -> tuple[str, str, int]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    with (
        stdout_path.open("w", buffering=1) as stdout_file,
        stderr_path.open("w", buffering=1) as stderr_file,
    ):
        stdout_file.write(f"[INFO] start={datetime.now()}\n")
        stdout_file.write(f"[INFO] command={' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def pump(stream, handle, sink: list[str], terminal) -> None:
            for line in iter(stream.readline, ""):
                sink.append(line)
                handle.write(line)
                handle.flush()
                terminal.write(line)
                terminal.flush()
            stream.close()

        stdout_thread = threading.Thread(
            target=pump,
            args=(proc.stdout, stdout_file, stdout_chunks, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump,
            args=(proc.stderr, stderr_file, stderr_chunks, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            returncode = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()

    return "".join(stdout_chunks), "".join(stderr_chunks), returncode


def build_row(
    trial_id: int,
    setting: str,
    params: dict[str, Any],
    stdout: str,
    stderr: str,
    elapsed_s: float | None,
    returncode: int,
    trial_dir: Path,
) -> dict[str, Any]:
    bacc = parse_percent_line(stdout, "Balanced Acc:")
    acc = parse_percent_line(stdout, "Accuracy:")
    task_bacc = parse_task_bacc(trial_dir / "results.csv")
    task_acc = parse_task_acc(stdout)
    status = "ok" if returncode == 0 and bacc is not None else "failed"
    row = {
        "trial_id": trial_id,
        "setting": setting,
        "mode": "naive",
        "status": status,
        "objective": "bacc_mean",
        "objective_value": bacc if bacc is not None else float("nan"),
        "bacc_mean": bacc if bacc is not None else float("nan"),
        "acc_mean": acc if acc is not None else float("nan"),
        "elapsed_s": elapsed_s if elapsed_s is not None else float("nan"),
        "returncode": returncode,
        "stdout_log": str(trial_dir / "stdout.log"),
        "stderr_log": str(trial_dir / "stderr.log"),
        "result_csv": str(trial_dir / "results.csv"),
        "stderr_tail": stderr[-600:],
        **params,
    }
    for task_id in range(6):
        row[f"task_{task_id}_bacc"] = task_bacc.get(
            task_id, float("nan")
        )
        row[f"task_{task_id}_acc"] = task_acc.get(
            task_id, float("nan")
        )
    return row


def run_trial(
    args: argparse.Namespace,
    trial_id: int,
    sampled_params: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    trial_dir = output_dir / f"trial_{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    result_csv = trial_dir / "results.csv"
    params = {**FIXED_PARAMS, **sampled_params}
    (trial_dir / "params.json").write_text(
        json.dumps(
            json_safe(
                {
                    "trial_id": trial_id,
                    "setting": args.setting,
                    "mode": "naive",
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
            trial_id, args.setting, params, stdout, "", None, 0, trial_dir
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
        "naive",
        "--result_csv",
        str(result_csv),
        "--efficiency_json",
        str(trial_dir / "efficiency.json"),
        *params_to_cli(params),
    ]

    print(f"\n[Trial {trial_id:04d}] {json.dumps(params, sort_keys=True)}")
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
    )
    print(
        f"[Trial {trial_id:04d}] status={row['status']} "
        f"bACC={row['bacc_mean']} elapsed={elapsed_s / 60:.1f}m"
    )
    return row


def metric(row: dict[str, Any]) -> float:
    try:
        return float(row["bacc_mean"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def ranked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if not math.isnan(metric(row))]
    return sorted(valid, key=metric, reverse=True)


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
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def append_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(json_safe(row))


def read_early_stop_signal(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"signal_path": str(path)}
    return payload if isinstance(payload, dict) else {"signal_path": str(path)}


def write_early_stop_signal(
    path: Path,
    row: dict[str, Any],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": "ood_bacc_threshold_reached",
        "created_at": datetime.now().isoformat(),
        "threshold_percent": threshold,
        "trial_id": int(row["trial_id"]),
        "bacc_mean_percent": float(row["bacc_mean"]),
        "params": {
            key: row[key]
            for key in [*FIXED_PARAMS.keys(), *SEARCH_SPACE.keys()]
        },
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2)
    except FileExistsError:
        return


def collect_worker_rows(output_root: Path, setting: str) -> list[dict[str, Any]]:
    paths = list(
        output_root.glob(f"gpu*_w*/{setting}/naive/summary_*.csv")
    )
    paths.extend(
        output_root.glob(f"{setting}/naive/summary_*.csv")
    )
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


def write_best(output_dir: Path, rows: list[dict[str, Any]], top_k: int) -> None:
    ranked = ranked_rows(rows)
    if not ranked:
        raise ValueError("No successful trial found.")
    best = dict(ranked[0])
    param_keys = [*FIXED_PARAMS.keys(), *SEARCH_SPACE.keys()]
    best_params = {key: best[key] for key in param_keys}
    best["params"] = best_params
    (output_dir / "best_trial.json").write_text(
        json.dumps(json_safe(best), indent=2), encoding="utf-8"
    )
    (output_dir / "best_config.json").write_text(
        json.dumps(json_safe(best_params), indent=2), encoding="utf-8"
    )
    write_rows(output_dir / "top_k.csv", ranked[:top_k])
    print(
        f"[Random Search] best trial={best['trial_id']} "
        f"bACC={best['bacc_mean']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random search for MergeSlide_TTA naive Class-IL."
    )
    parser.add_argument("--setting", choices=["ind", "ood"], default="ind")
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--merge_dir", required=True)
    parser.add_argument("--finetuned_dir", required=True)
    parser.add_argument("--output_dir", default="./logs/tune_naive")
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
    parser.add_argument(
        "--early_stop_bacc",
        type=float,
        default=0.0,
        help=(
            "Stop scheduling new OOD trials after a successful trial reaches "
            "this mean bACC percentage. Disabled for IND and values <= 0."
        ),
    )
    parser.add_argument(
        "--early_stop_signal",
        default="",
        help="Shared JSON stop-signal path used by parallel OOD workers.",
    )
    args = parser.parse_args()
    if args.n_steps < 1:
        parser.error("--n_steps must be >= 1")
    if args.early_stop_bacc < 0:
        parser.error("--early_stop_bacc must be >= 0")
    args.timeout_sec = args.timeout_sec or None
    return args


def main() -> None:
    args = parse_args()
    FIXED_PARAMS["n_steps"] = args.n_steps
    output_root = Path(args.output_dir)
    output_dir = output_root / args.setting / "naive"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else output_dir
        / f"manifest_{args.setting}_naive_{args.n_trials}_{args.seed}.json"
    )

    print(
        f"[Random Search] mode=naive setting={args.setting} "
        f"trials={args.n_trials}"
    )
    print(f"[Random Search] fixed={FIXED_PARAMS}")
    print(f"[Random Search] space={SEARCH_SPACE}")

    if args.prepare_manifest:
        manifest = build_manifest(args.n_trials, args.seed)
        save_manifest(manifest, manifest_path)
        print(f"[Random Search] manifest={manifest_path}")
        return

    if args.summarize_only:
        rows = collect_worker_rows(output_root, args.setting)
        if not rows:
            raise FileNotFoundError(
                f"No worker summaries found under {output_root}"
            )
        write_rows(output_dir / "summary_all.csv", rows)
        write_best(output_dir, rows, args.top_k)
        return

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Prepare the manifest first."
        )

    manifest = load_manifest(manifest_path)
    early_stop_enabled = args.setting == "ood" and args.early_stop_bacc > 0
    early_stop_signal = (
        Path(args.early_stop_signal)
        if early_stop_enabled and args.early_stop_signal
        else None
    )
    if early_stop_enabled and early_stop_signal is None:
        early_stop_signal = manifest_path.with_suffix(".early_stop.json")
    if early_stop_enabled:
        print(
            f"[Random Search] OOD early stop: "
            f"bACC>={args.early_stop_bacc:.4f}% "
            f"signal={early_stop_signal}"
        )
    trial_end = (
        args.trial_end if args.trial_end is not None else args.n_trials
    )
    summary_path = output_dir / (
        f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
    )
    rows: list[dict[str, Any]] = []
    for item in manifest[args.trial_start:trial_end]:
        stop_payload = read_early_stop_signal(early_stop_signal)
        if stop_payload is not None:
            print(
                "[Random Search] stopping before next trial because the "
                f"shared OOD threshold signal exists: {stop_payload}"
            )
            break
        row = run_trial(
            args,
            int(item["trial_id"]),
            dict(item["params"]),
            output_dir,
        )
        rows.append(row)
        row["early_stop_bacc_percent"] = (
            args.early_stop_bacc if early_stop_enabled else float("nan")
        )
        row["early_stop_hit"] = bool(
            early_stop_enabled
            and row.get("status") == "ok"
            and math.isfinite(metric(row))
            and metric(row) >= args.early_stop_bacc
        )
        append_row(summary_path, row)
        if row["early_stop_hit"]:
            write_early_stop_signal(
                early_stop_signal,
                row,
                args.early_stop_bacc,
            )
            print(
                f"[Random Search] EARLY STOP: trial={row['trial_id']} "
                f"bACC={metric(row):.4f}% >= "
                f"{args.early_stop_bacc:.4f}%"
            )
            break

    print(f"[Random Search] worker summary={summary_path}")
    if ranked_rows(rows):
        best = ranked_rows(rows)[0]
        print(
            f"[Random Search] worker best trial={best['trial_id']} "
            f"bACC={best['bacc_mean']}"
        )


if __name__ == "__main__":
    main()
