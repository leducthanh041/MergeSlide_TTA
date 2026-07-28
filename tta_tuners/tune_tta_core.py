"""
tta_tuners/tune_tta_core.py
=============================
Random search cho MergeSlide-TTA-Unified — TỰ CHỨA, không phụ thuộc
tune_tta_v3.py (đã xoá cùng đợt hợp nhất kiến trúc).

Search space CHỈ còn Module F percentile — n_steps (Module G) và
regularizer_type (Module H) ĐÃ CHỐT qua thực nghiệm (n_steps=3, div_n,
swag), không sweep lại trừ khi có ablation cụ thể (dùng configs/ablation/
cho việc đó, không dùng tuner này).

Usage::
    # Bước 1: chuẩn bị manifest (chạy 1 lần)
    python tta_tuners/tune_tta_core.py --prepare_manifest --n_trials 20 --seed 42 \\
        --mode naive --base_config configs/default_tta_core_eval_num_workers0.yaml \\
        --merge_dir ./checkpoints/merged --swag_dir ./checkpoints/swag_diagonal \\
        --output_dir ./logs/tune_tta_core

    # Bước 2: chạy các trial
    python tta_tuners/tune_tta_core.py --n_trials 20 --seed 42 --mode naive \\
        --base_config configs/default_tta_core_eval_num_workers0.yaml \\
        --merge_dir ./checkpoints/merged --swag_dir ./checkpoints/swag_diagonal \\
        --output_dir ./logs/tune_tta_core --num_folds 10
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


SEARCH_SPACE: dict[str, list] = {
    "conf_percentile":  [0.3, 0.4, 0.5, 0.6, 0.7, 1.0],   # 1.0 = tat S_conf
    "agree_percentile": [0.3, 0.4, 0.5, 0.6, 0.7, 1.0],   # 1.0 = tat S_agree
    "conf_window":      [100, 300, 500],
    "agree_window":     [100, 300, 500],
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (tu chua, khong import tu file da xoa)
# ──────────────────────────────────────────────────────────────────────────────

def run_command_streaming(
    cmd: list[str], cwd: str, stdout_path: Path, stderr_path: Path,
    header_lines: list[str], timeout_sec: int = 7200,
) -> tuple[str, str, int]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        for line in header_lines:
            out_f.write(line + "\n")
        out_f.flush()
        proc = subprocess.run(
            cmd, cwd=cwd, stdout=out_f, stderr=err_f,
            timeout=timeout_sec, text=True,
        )
    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    return stdout, stderr, proc.returncode


def parse_bacc_from_output(stdout: str) -> float | None:
    """Tim dong 'Balanced Acc:    XX.XXXX% (...)' trong log."""
    m = re.search(r"Balanced Acc:\s*([\d.]+)%", stdout)
    return float(m.group(1)) if m else None


def parse_bacc_per_task(stdout: str) -> dict[int, float]:
    out = {}
    for m in re.finditer(r"Task (\d+):\s*([\d.]+)%", stdout):
        out[int(m.group(1))] = float(m.group(2))
    return out


def save_manifest(manifest: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def has_cached_trial_result(result_csv: Path) -> bool:
    return result_csv.exists() and result_csv.stat().st_size > 0


def _row_metric(row: dict, key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def collect_worker_summaries(output_dir: Path) -> list[dict]:
    rows = []
    for summary_csv in sorted(output_dir.glob("summary_*.csv")):
        with open(summary_csv) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def write_combined_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Core tuning logic
# ──────────────────────────────────────────────────────────────────────────────

def sample_config(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def build_manifest(n_trials: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    manifest = []
    for trial_id in range(n_trials):
        manifest.append({"trial_id": trial_id, "params": sample_config(rng)})
    return manifest


def make_trial_config(base_config_path: str, params: dict, trial_dir: Path) -> str:
    with open(base_config_path) as f:
        config = yaml.safe_load(f) or {}
    config.setdefault("tta", {})
    for key, value in params.items():
        config["tta"][key] = value
    trial_config_path = trial_dir / "trial_config.yaml"
    with open(trial_config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    return str(trial_config_path)


def run_one_trial(
    trial_id: int, params: dict, mode: str, base_config: str,
    merge_dir: str, swag_dir: str, output_dir: Path, python_bin: str,
    project_root: str, num_folds: int, entrypoint_wrapper: str | None,
) -> dict:
    trial_dir = output_dir / f"trial_{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_config_path = make_trial_config(base_config, params, trial_dir)
    result_csv = str(trial_dir / "results.csv")
    log_out, log_err = trial_dir / "stdout.log", trial_dir / "stderr.log"

    with open(trial_dir / "params.json", "w") as f:
        json.dump({"trial_id": trial_id, "mode": mode, **params}, f, indent=2)

    cmd = [python_bin, "-u", entrypoint_wrapper or "test_tta_core.py"]
    if entrypoint_wrapper:
        cmd += ["--entrypoint", "test_tta_core.py"]
    cmd += [
        "--config", trial_config_path, "--mode", mode,
        "--merge_model_path", merge_dir, "--swag_dir", swag_dir,
        "--result_csv", result_csv,
        "--fold_start", "0", "--fold_end", str(num_folds),
    ]

    print(f"\n{'=' * 60}\n[Trial {trial_id:04d}] mode={mode} objective=bACC")
    for key, value in params.items():
        print(f"    {key:18s} = {value}")
    print(f"  config -> {trial_config_path}\n{'=' * 60}")

    t0 = time.time()
    header = [
        f"[INFO] start at {datetime.now()}", f"[INFO] command={' '.join(cmd)}",
        f"[INFO] trial_id={trial_id} mode={mode}",
        f"[INFO] params={json.dumps(params, sort_keys=True)}",
    ]
    try:
        stdout, stderr, rc = run_command_streaming(
            cmd, project_root, Path(log_out), Path(log_err), header, timeout_sec=7200,
        )
    except Exception as exc:
        stdout, stderr, rc = "", f"STREAMING_ERROR: {exc}", -1

    elapsed = time.time() - t0
    bacc = parse_bacc_from_output(stdout)
    task_baccs = parse_bacc_per_task(stdout)
    status = "ok" if rc == 0 and bacc is not None else "failed"

    result = {
        "trial_id": trial_id, "mode": mode, "status": status,
        "bacc_mean": bacc if bacc is not None else float("nan"),
        "elapsed_s": elapsed, "returncode": rc,
        **{f"task_{t}_bacc": task_baccs.get(t, float("nan")) for t in range(6)},
        **params,
    }
    if bacc is not None:
        print(f"[Trial {trial_id:04d}] -> bACC = {bacc:.4f}% ({elapsed / 60:.1f} min)")
    else:
        print(f"[Trial {trial_id:04d}] -> FAILED (rc={rc})\n  stderr: {stderr[-400:]}")
    return result


def print_best(results: list[dict], n: int = 5) -> None:
    valid = [r for r in results if not np.isnan(_row_metric(r, "bacc_mean"))]
    if not valid:
        print("  Chua co trial thanh cong."); return
    top = sorted(valid, key=lambda r: _row_metric(r, "bacc_mean"), reverse=True)[:n]
    print(f"\n{'-' * 60}\nTOP {min(n, len(top))} TRIALS (objective=bACC):")
    for idx, row in enumerate(top):
        print(f"  #{idx + 1}  bACC={float(row.get('bacc_mean', 0)):.4f}%  trial={int(row['trial_id']):04d}")
        for key in SEARCH_SPACE.keys():
            print(f"       {key:18s} = {row.get(key, 'N/A')}")
    print(f"{'-' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Random search Module F percentile cho MergeSlide-TTA-Unified")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--mode", type=str, default="naive", choices=["naive", "tcp"])
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--merge_dir", type=str, required=True)
    parser.add_argument("--swag_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./logs/tune_tta_core")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--python_bin", type=str, default=None)
    parser.add_argument("--entrypoint_wrapper", type=str, default="tools/run_classil_with_pt_features.py")
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--prepare_manifest", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_end", type=int, default=None)
    args = parser.parse_args()

    if args.python_bin:
        python_bin = args.python_bin
    else:
        default_py = "/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
        python_bin = default_py if Path(default_py).exists() else sys.executable

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_start = args.trial_start
    trial_end = args.trial_end if args.trial_end is not None else args.n_trials
    manifest_path = Path(
        args.manifest_path or output_dir / f"manifest_{args.mode}_{args.n_trials}_{args.seed}.json"
    )
    summary_csv = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"[Tune] mode={args.mode} objective=bACC n_trials={args.n_trials} "
          f"seed={args.seed} folds={args.num_folds}")
    print(f"[Tune] base_config={args.base_config}  output_dir={output_dir}")
    print("Search space:")
    for key, value in SEARCH_SPACE.items():
        print(f"  {key:18s}: {value}")

    if args.summarize_only:
        rows = collect_worker_summaries(output_dir)
        write_combined_summary(rows, output_dir / "summary_all.csv")
        print_best(rows, n=10)
        return

    if args.prepare_manifest:
        manifest = build_manifest(args.n_trials, args.seed)
        save_manifest(manifest, manifest_path)
        print(f"[Tune] manifest written -> {manifest_path}")
        return

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Chay voi --prepare_manifest truoc.")

    manifest = load_manifest(manifest_path)
    all_results: list[dict] = []
    fieldnames = None

    for item in manifest[trial_start:trial_end]:
        trial_id = int(item["trial_id"])
        params = dict(item["params"])
        result_csv = output_dir / f"trial_{trial_id:04d}" / "results.csv"

        if has_cached_trial_result(result_csv):
            print(f"[Trial {trial_id:04d}] SKIP cached -> {result_csv}")
            continue

        result = run_one_trial(
            trial_id=trial_id, params=params, mode=args.mode, base_config=args.base_config,
            merge_dir=args.merge_dir, swag_dir=args.swag_dir, output_dir=output_dir,
            python_bin=python_bin, project_root=args.project_root, num_folds=args.num_folds,
            entrypoint_wrapper=args.entrypoint_wrapper,
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

        print_best(all_results, n=5)

    print(f"\n{'=' * 60}\nTUNING COMPLETE - {args.n_trials} trials - mode={args.mode}")
    print(f"Summary -> {summary_csv}")
    print_best(all_results, n=10)


if __name__ == "__main__":
    main()
