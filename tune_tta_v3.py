"""
tune_tta_v3.py — Random Search hyperparameter tuning cho MergeSlide-TTA v3.

Hai phase tuning:
  Phase 1 (routing): maximize routing_acc — tune margin_task, gamma_task, delta_margin, alpha_task_prompt, tau_task
  Phase 2 (class):   maximize bACC        — tune eta_base, delta, tau_c, gamma_class

Cơ chế: mỗi trial tạo một yaml config tạm (copy từ base config, override
section tta), truyền thẳng vào --config của test_tta.py.
KHÔNG cần patch test_tta.py.

Usage:
    python tune_tta.py \\
        --n_trials 30 \\
        --setting ind \\
        --base_config configs/default_tta_eval_num_workers0.yaml \\
        --merge_dir   ./checkpoints/merged \\
        --swag_dir    ./checkpoints/swag_diagonal \\
        --finetuned_dir ./checkpoints/finetuned \\
        --output_dir  ./logs/tune_tta

    python tune_tta.py \\
        --n_trials 30 \\
        --setting ood \\
        --base_config configs/default_tta_ood_eval_num_workers0.yaml \\
        --merge_dir   ./checkpoints_ood/merged \\
        --swag_dir    ./checkpoints_ood/swag_diagonal \\
        --finetuned_dir ./checkpoints_ood/finetuned \\
        --output_dir  ./logs/tune_tta_ood
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Search space — tham số quan trọng bậc 1 và bậc 2
# ──────────────────────────────────────────────────────────────────────────────

# Search space theo phase:
#   phase="routing" → SEARCH_SPACE_ROUTING
#   phase="class"   → SEARCH_SPACE_CLASS
SEARCH_SPACE_ROUTING: dict[str, list] = {
    # Primary: L_task margin và weight
    "margin_task":        [0.5, 1.0, 2.0, 3.0, 5.0],
    "gamma_task":         [0.5, 1.0, 2.0, 3.0],
    # Task prompt EMA gate
    "delta_margin":       [0.05, 0.10, 0.15, 0.20, 0.30],
    "alpha_task_prompt":  [0.99, 0.995, 0.999],
    # TCP inference gate
    "tau_task":           [0.50, 0.60, 0.70, 0.80],
}

SEARCH_SPACE_CLASS: dict[str, list] = {
    # Giữ nguyên từ tune_tta.py gốc — tune sau khi routing params đã fixed
    "eta_base":    [1e-5, 5e-5, 1e-4, 3e-4, 5e-4],
    "delta":       [0.01, 0.03, 0.05, 0.10, 0.20],
    "tau_c":       [0.05, 0.07, 0.10, 0.20, 0.30],
    "tau_ood":     [0.20, 0.30, 0.50, 0.70],
    "gamma_class": [0.5,  1.0,  2.0],
}

# Được set tại runtime theo --phase argument
SEARCH_SPACE: dict[str, list] = SEARCH_SPACE_ROUTING  # default

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sample_config(rng: random.Random) -> dict:
    """Sample một bộ tham số ngẫu nhiên từ SEARCH_SPACE."""
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def load_default_tta_params(base_config_path: str) -> dict:
    """Load the default tta block from the base config."""
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    raw_tta = base_cfg.get("tta", {}) if isinstance(base_cfg, dict) else {}
    return {k: raw_tta.get(k) for k in SEARCH_SPACE.keys() if k in raw_tta}


def build_manifest(
    n_trials: int,
    seed: int,
    excluded_params: dict | None = None,
) -> list[dict]:
    """Build one deterministic manifest for all trials, skipping excluded params."""
    rng = random.Random(seed)
    manifest: list[dict] = []
    trial_id = 0
    excluded_count = 0
    while len(manifest) < n_trials:
        params = sample_config(rng)
        if excluded_params and all(params.get(k) == excluded_params.get(k) for k in SEARCH_SPACE.keys()):
            excluded_count += 1
            continue
        manifest.append({"trial_id": trial_id, "params": params})
        trial_id += 1
    if excluded_count:
        print(f"[TTA Tuning] excluded {excluded_count} default-config samples")
    return manifest


def save_manifest(manifest: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise ValueError(f"Invalid manifest format: {manifest_path}")
    return manifest


def make_trial_config(
    base_config_path: str,
    params: dict,
    trial_dir: Path,
) -> str:
    """
    Tạo yaml config tạm cho trial bằng cách:
      1. Load base config (yaml thuần — bỏ OmegaConf interpolation)
      2. Override section tta với params của trial
      3. Ghi vào trial_dir/trial_config.yaml
    Trả về path tới file yaml tạm.
    """
    # Load base config dưới dạng text để giữ nguyên cấu trúc
    with open(base_config_path, "r") as f:
        base_text = f.read()

    # Parse bằng yaml (không resolve interpolation — giữ nguyên ${...})
    # Dùng safe_load để đọc, sau đó override tta section
    config = yaml.safe_load(base_text)

    # Override từng tham số được tune
    if "tta" not in config:
        config["tta"] = {}
    for k, v in params.items():
        config["tta"][k] = v

    # Ghi ra file tạm
    trial_config_path = trial_dir / "trial_config.yaml"
    with open(trial_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return str(trial_config_path)


def parse_bacc_from_output(stdout: str) -> float | None:
    """
    Parse bACC từ dòng: "Balanced Acc:    XX.XXXX% (XX.XXXX%)"
    Trả về float (0–100) hoặc None nếu không tìm thấy.
    """
    for line in stdout.splitlines():
        if "Balanced Acc:" in line:
            try:
                part = line.split("Balanced Acc:")[1].strip()
                mean_str = part.split("%")[0].strip()
                return float(mean_str)
            except (IndexError, ValueError):
                continue
    return None


def parse_bacc_per_task(stdout: str) -> dict[int, float]:
    """
    Parse bACC per task từ dòng: "  Task X: XX.XXXX% (XX.XXXX%)"
    """
    task_baccs: dict[int, float] = {}
    in_acc_section = False
    for line in stdout.splitlines():
        if "Acc per task:" in line:
            in_acc_section = True
            continue
        if in_acc_section:
            s = line.strip()
            if s.startswith("Task "):
                try:
                    task_id  = int(s.split(":")[0].replace("Task", "").strip())
                    val_str  = s.split(":")[1].strip().split("%")[0].strip()
                    task_baccs[task_id] = float(val_str)
                except (IndexError, ValueError):
                    continue
            elif task_baccs and not s.startswith("Task"):
                break
    return task_baccs

def parse_routing_acc_from_output(stdout: str) -> float | None:
    """
    Parse mean routing accuracy từ dòng:
    "Routing Accuracy (mean): XX.XXXX% (XX.XXXX%)"
    Trả về float (0–100) hoặc None nếu không tìm thấy.
    """
    for line in stdout.splitlines():
        if "Routing Accuracy (mean):" in line:
            try:
                part = line.split("Routing Accuracy (mean):")[1].strip()
                mean_str = part.split("%")[0].strip()
                return float(mean_str)
            except (IndexError, ValueError):
                continue
    return None


def parse_routing_acc_per_task(stdout: str) -> dict[int, float]:
    """
    Parse routing accuracy per task từ dòng:
    "  Routing Task X: XX.XXXX% (XX.XXXX%)"
    """
    task_raccs: dict[int, float] = {}
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("Routing Task "):
            try:
                task_id  = int(s.split(":")[0].replace("Routing Task", "").strip())
                val_str  = s.split(":")[1].strip().split("%")[0].strip()
                task_raccs[task_id] = float(val_str)
            except (IndexError, ValueError):
                continue
    return task_raccs




def run_one_trial(
    trial_id:        int,
    params:          dict,
    setting:         str,
    phase:           str,
    base_config:     str,
    merge_dir:       str,
    swag_dir:        str,
    finetuned_dir:   str,
    output_dir:      Path,
    python_bin:      str,
    project_root:    str,
    num_folds:       int,
    entrypoint_wrapper: str | None,
) -> dict:
    """
    Chạy một trial: tạo yaml config tạm → chạy test_tta.py → parse kết quả.
    """
    trial_dir = output_dir / f"trial_{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tạo yaml config tạm
    trial_config_path = make_trial_config(base_config, params, trial_dir)

    # 2. Paths cho output của trial này
    result_csv = str(trial_dir / "results.csv")
    log_out    = trial_dir / "stdout.log"
    log_err    = trial_dir / "stderr.log"

    # Lưu params để tra cứu sau
    with open(trial_dir / "params.json", "w") as f:
        json.dump({"trial_id": trial_id, "setting": setting, **params}, f, indent=2)

    # 3. Build command — prefer PT-first wrapper to avoid direct H5 reads.
    if entrypoint_wrapper:
        cmd = [
            python_bin, "-u", entrypoint_wrapper,
            "--entrypoint",       "test_tta_v3.py",
            "--config",           trial_config_path,
            "--save_dir",         finetuned_dir,
            "--merge_model_path", merge_dir,
            "--swag_dir",         swag_dir,
            "--mode",             "classil_tcp",
            "--result_csv",       result_csv,
            "--fold_start",       "0",
            "--fold_end",         str(num_folds),
        ]
    else:
        cmd = [
            python_bin, "-u", "test_tta_v3.py",
            "--config",           trial_config_path,
            "--save_dir",         finetuned_dir,
            "--merge_model_path", merge_dir,
            "--swag_dir",         swag_dir,
            "--mode",             "classil_tcp",
            "--result_csv",       result_csv,
            "--fold_start",       "0",
            "--fold_end",         str(num_folds),
        ]

    print(f"\n{'='*60}")
    print(f"[Trial {trial_id:04d}] Setting={setting}")
    for k, v in params.items():
        print(f"    {k:15s} = {v}")
    print(f"  config → {trial_config_path}")
    print(f"{'='*60}")

    # 4. Chạy subprocess
    t0 = time.time()
    header_lines = [
        f"[INFO] start at {datetime.now()}",
        f"[INFO] command={' '.join(cmd)}",
        f"[INFO] trial_id={trial_id} setting={setting}",
        f"[INFO] params={json.dumps(params, sort_keys=True)}",
    ]
    try:
        stdout, stderr, returncode = run_command_streaming(
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

    # 6. Parse kết quả
    bacc          = parse_bacc_from_output(stdout)
    task_baccs    = parse_bacc_per_task(stdout)
    routing_acc   = parse_routing_acc_from_output(stdout)
    task_routing  = parse_routing_acc_per_task(stdout)

    # Objective phụ thuộc vào phase
    if phase == "routing":
        primary_metric = routing_acc
        primary_label  = "routing_acc"
    else:
        primary_metric = bacc
        primary_label  = "bACC"

    status = "ok" if returncode == 0 and primary_metric is not None else "failed"

    result = {
        "trial_id":      trial_id,
        "setting":       setting,
        "phase":         phase,
        "status":        status,
        "bacc_mean":     bacc         if bacc         is not None else float("nan"),
        "routing_acc":   routing_acc  if routing_acc  is not None else float("nan"),
        "elapsed_s":     elapsed,
        "returncode":    returncode,
        **{f"task_{t}_bacc":         task_baccs.get(t,   float("nan")) for t in range(6)},
        **{f"task_{t}_routing_acc":  task_routing.get(t, float("nan")) for t in range(6)},
        **params,
    }

    if primary_metric is not None:
        print(f"[Trial {trial_id:04d}] → {primary_label} = {primary_metric:.4f}%  ({elapsed/60:.1f} min)")
    else:
        print(f"[Trial {trial_id:04d}] → FAILED (rc={returncode})")
        print(f"  stderr: {stderr[-400:]}")

    return result


def write_manifest_csv(manifest: list[dict], manifest_csv_path: Path) -> None:
    """Write a human-readable CSV companion to the JSON manifest."""
    with open(manifest_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", *SEARCH_SPACE.keys()])
        writer.writeheader()
        for item in manifest:
            writer.writerow({"trial_id": item["trial_id"], **item["params"]})


def has_cached_trial_result(result_csv: Path) -> bool:
    """Return True if a trial result CSV already exists and has at least one row."""
    if not result_csv.exists() or result_csv.stat().st_size == 0:
        return False
    try:
        with open(result_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            return any(True for _ in reader)
    except Exception:
        return False


def load_trial_result_row(result_csv: Path) -> dict | None:
    """Load the first cached row from a completed trial result CSV."""
    if not has_cached_trial_result(result_csv):
        return None
    try:
        with open(result_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
    except Exception:
        return None
    if not row:
        return None

    for key in ("trial_id",):
        if key in row and row[key] not in ("", None):
            try:
                row[key] = int(float(row[key]))
            except ValueError:
                pass
    for key in ("bacc_mean", "routing_acc", "elapsed_s"):
        if key in row and row[key] not in ("", None):
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
    if "returncode" in row and row["returncode"] not in ("", None):
        try:
            row["returncode"] = int(float(row["returncode"]))
        except ValueError:
            pass
    for key in SEARCH_SPACE:
        if key in row and row[key] not in ("", None):
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
    for key in list(row.keys()):
        if key.startswith("task_") and (key.endswith("_bacc") or key.endswith("_routing_acc")) and row[key] not in ("", None):
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
    return row


def run_command_streaming(
    cmd: list[str],
    cwd: str,
    stdout_path: Path,
    stderr_path: Path,
    header_lines: list[str] | None = None,
    timeout_sec: int | None = None,
) -> tuple[str, str, int]:
    """Run command while streaming stdout/stderr to files and current console."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stdout_path, "w", buffering=1) as stdout_file, open(stderr_path, "w", buffering=1) as stderr_file:
        if header_lines:
            for line in header_lines:
                stdout_file.write(line.rstrip("\n") + "\n")
                stderr_file.write(line.rstrip("\n") + "\n")
            stdout_file.flush()
            stderr_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def pump(stream, file_handle, sink, is_err: bool) -> None:
            for line in iter(stream.readline, ""):
                sink.append(line)
                file_handle.write(line)
                file_handle.flush()
                if is_err:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            stream.close()

        stdout_thread = threading.Thread(
            target=pump, args=(proc.stdout, stdout_file, stdout_chunks, False), daemon=True
        )
        stderr_thread = threading.Thread(
            target=pump, args=(proc.stderr, stderr_file, stderr_chunks, True), daemon=True
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


def collect_worker_summaries(root_output_dir: Path) -> list[dict]:
    """Collect latest per-trial rows from all worker summary csv files."""
    summary_files = sorted(
        root_output_dir.glob("*/summary_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not summary_files:
        raise FileNotFoundError(
            f"No worker summary files found under {root_output_dir}"
        )

    merged: dict[int, dict] = {}
    for csv_path in summary_files:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                trial_raw = row.get("trial_id", "")
                try:
                    trial_id = int(float(trial_raw))
                except (TypeError, ValueError):
                    continue
                row["trial_id"] = trial_id
                for key in ("bacc_mean", "routing_acc", "elapsed_s"):
                    if key in row and row[key] not in ("", None):
                        try:
                            row[key] = float(row[key])
                        except ValueError:
                            pass
                if "returncode" in row and row["returncode"] not in ("", None):
                    try:
                        row["returncode"] = int(float(row["returncode"]))
                    except ValueError:
                        pass
                for key in SEARCH_SPACE:
                    if key in row and row[key] not in ("", None):
                        try:
                            row[key] = float(row[key])
                        except ValueError:
                            pass
                for key in row:
                    if key.startswith("task_") and (key.endswith("_bacc") or key.endswith("_routing_acc")) and row[key] not in ("", None):
                        try:
                            row[key] = float(row[key])
                        except ValueError:
                            pass
                merged[trial_id] = row

    rows = []
    for trial_id in sorted(merged):
        row = dict(merged[trial_id])
        row["trial_id"] = trial_id
        rows.append(row)
    return rows


def write_combined_summary(rows: list[dict], out_csv: Path) -> None:
    """Write a single merged summary CSV for all trials."""
    if not rows:
        raise ValueError("No rows to write in combined summary.")
    fieldnames = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_metric(row: dict, metric_key: str) -> float:
    try:
        return float(row.get(metric_key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def write_best_trial_artifacts(rows: list[dict], out_dir: Path, phase: str, top_k: int = 5) -> None:
    """Write best_trial.json and top_k.csv from merged summary rows."""
    metric_key = "routing_acc" if phase == "routing" else "bacc_mean"
    valid = [r for r in rows if not np.isnan(_row_metric(r, metric_key))]
    if not valid:
        raise ValueError("No valid trials found for best-trial artifacts.")

    ranked = sorted(valid, key=lambda r: _row_metric(r, metric_key), reverse=True)
    best = ranked[0]

    best_json = out_dir / "best_trial.json"
    best_json.parent.mkdir(parents=True, exist_ok=True)
    with open(best_json, "w") as f:
        json.dump(best, f, indent=2)

    top_rows = ranked[:top_k]
    top_csv = out_dir / "top_k.csv"
    fieldnames = list(top_rows[0].keys())
    with open(top_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"[TTA Tuning] best trial ({metric_key}) → {best_json}")
    print(f"[TTA Tuning] top-{len(top_rows)} trials ({metric_key}) → {top_csv}")


def print_best(results: list[dict], n: int = 5, phase: str = "routing") -> None:
    if phase == "routing":
        metric_key = "routing_acc"
        metric_label = "routing_acc"
    else:
        metric_key = "bacc_mean"
        metric_label = "bACC"
    valid = [r for r in results if not np.isnan(float(r.get(metric_key, float("nan"))))]
    if not valid:
        print("  Chưa có trial thành công.")
        return
    top = sorted(valid, key=lambda r: float(r.get(metric_key, 0.0)), reverse=True)[:n]
    print(f"\n{'─'*60}")
    print(f"TOP {min(n, len(top))} TRIALS  (phase={phase}, objective={metric_label}):")
    for i, r in enumerate(top):
        print(f"  #{i+1}  {metric_label}={float(r.get(metric_key,0)):.4f}%  bACC={float(r.get('bacc_mean',0)):.4f}%  trial={r['trial_id']:04d}")
        for k in SEARCH_SPACE:
            print(f"       {k:15s} = {r.get(k, 'N/A')}")
    print(f"{'─'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random search TTA hyperparameter tuning"
    )
    parser.add_argument("--n_trials",       type=int, default=30)
    parser.add_argument("--setting",        type=str, default="ind",
                        choices=["ind", "ood"])
    parser.add_argument("--base_config",    type=str, required=True,
                        help="Base yaml config (default_tta_eval_num_workers0.yaml)")
    parser.add_argument("--merge_dir",      type=str, required=True)
    parser.add_argument("--swag_dir",       type=str, required=True)
    parser.add_argument("--finetuned_dir",  type=str, required=True)
    parser.add_argument("--output_dir",     type=str, default="./logs/tune_tta")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--num_folds",      type=int, default=10)
    parser.add_argument("--project_root",   type=str, default=".")
    parser.add_argument("--python_bin",     type=str, default=None)
    parser.add_argument("--entrypoint_wrapper", type=str, default=None,
                        help="Optional PT-first wrapper, e.g. tools/run_classil_with_pt_features.py")
    parser.add_argument("--phase", type=str, default="routing",
                        choices=["routing", "class"],
                        help="routing: optimize routing_acc (tune margin_task/gamma_task/...); "
                             "class: optimize bACC (tune eta_base/delta/tau_c/...)")
    parser.add_argument("--manifest_path",  type=str, default=None,
                        help="Shared manifest JSON path for all trials.")
    parser.add_argument("--prepare_manifest", action="store_true",
                        help="Build manifest and exit without running trials.")
    parser.add_argument("--summarize_only", action="store_true",
                        help="Merge worker summary files and exit.")
    parser.add_argument("--trial_start",    type=int, default=0,
                        help="Inclusive start index for this worker slice.")
    parser.add_argument("--trial_end",      type=int, default=None,
                        help="Exclusive end index for this worker slice.")
    args = parser.parse_args()

    # Set SEARCH_SPACE theo phase
    global SEARCH_SPACE
    if args.phase == "routing":
        SEARCH_SPACE = SEARCH_SPACE_ROUTING
        print(f"[TTA v3 Tuning] Phase: ROUTING — objective = routing_acc")
    else:
        SEARCH_SPACE = SEARCH_SPACE_CLASS
        print(f"[TTA v3 Tuning] Phase: CLASS — objective = bACC")

    # Python binary
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
        else output_dir / f"manifest_{args.setting}_{args.n_trials}_{args.seed}.json"
    )

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = output_dir / f"summary_{timestamp}.csv"

    print(f"[TTA Tuning] Setting: {args.setting.upper()}")
    print(f"[TTA Tuning] base_config: {args.base_config}")
    print(f"[TTA Tuning] n_trials={args.n_trials}  seed={args.seed}  folds={args.num_folds}")
    print(f"[TTA Tuning] trial_start={trial_start}  trial_end={trial_end}")
    print(f"[TTA Tuning] output_dir={output_dir}")
    print(f"[TTA Tuning] manifest_path={manifest_path}")
    print(f"[TTA Tuning] entrypoint_wrapper={args.entrypoint_wrapper or 'disabled'}")
    print(f"\nSearch space:")
    for k, v in SEARCH_SPACE.items():
        print(f"  {k:15s}: {v}")

    if args.summarize_only:
        combined_rows = collect_worker_summaries(output_dir)
        combined_csv = output_dir / "summary_all.csv"
        write_combined_summary(combined_rows, combined_csv)
        write_best_trial_artifacts(combined_rows, output_dir, phase=args.phase, top_k=5)
        print(f"[TTA Tuning] combined summary → {combined_csv}")
        print_best(combined_rows, n=10)
        return

    if args.prepare_manifest:
        excluded_params = load_default_tta_params(args.base_config)
        manifest = build_manifest(args.n_trials, args.seed, excluded_params=excluded_params)
        save_manifest(manifest, manifest_path)
        write_manifest_csv(manifest, manifest_path.with_suffix(".csv"))
        print(f"[TTA Tuning] manifest written → {manifest_path}")
        print(f"[TTA Tuning] manifest csv → {manifest_path.with_suffix('.csv')}")
        return

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run once with --prepare_manifest before launching workers."
        )

    manifest = load_manifest(manifest_path)
    if len(manifest) < args.n_trials:
        raise ValueError(
            f"Manifest has {len(manifest)} items but n_trials={args.n_trials}"
        )

    all_results: list[dict] = []
    fieldnames = None

    for item in manifest[trial_start:trial_end]:
        trial_id = int(item["trial_id"])
        params = dict(item["params"])
        trial_dir = output_dir / f"trial_{trial_id:04d}"
        result_csv = trial_dir / "results.csv"

        if has_cached_trial_result(result_csv):
            print(f"[Trial {trial_id:04d}] SKIP cached result → {result_csv}")
            cached_row = load_trial_result_row(result_csv)
            if cached_row is not None:
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

        result = run_one_trial(
            trial_id       = trial_id,
            params         = params,
            setting        = args.setting,
            phase          = args.phase,
            base_config    = args.base_config,
            merge_dir      = args.merge_dir,
            swag_dir       = args.swag_dir,
            finetuned_dir  = args.finetuned_dir,
            output_dir     = output_dir,
            python_bin     = python_bin,
            project_root   = args.project_root,
            num_folds      = args.num_folds,
            entrypoint_wrapper = args.entrypoint_wrapper,
        )
        all_results.append(result)

        # Append vào summary CSV sau mỗi trial — không mất data nếu bị interrupt
        if fieldnames is None:
            fieldnames = list(result.keys())
        write_header = not summary_csv.exists()
        with open(summary_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result)

        print_best(all_results, n=5, phase=args.phase)

    print(f"\n{'='*60}")
    print(f"TUNING COMPLETE — {args.n_trials} trials — {args.setting.upper()}")
    print(f"Summary → {summary_csv}")
    print_best(all_results, n=10, phase=args.phase)


if __name__ == "__main__":
    main()
