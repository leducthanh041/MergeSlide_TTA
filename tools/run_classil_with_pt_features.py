#!/usr/bin/env python3
"""Run an existing CLASS-IL entrypoint with PT-first feature loading.

This wrapper leaves the original evaluation code untouched. It patches the
dataset classes at runtime so feature tensors are read from pt_files when
available, while coordinates still come from the existing H5 files.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import h5py
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mergeslide_tta.constants import TOTAL_CLASSES


DEFAULT_ENTRYPOINT = PROJECT_ROOT / "test_classIL_task_prompt.py"
GLOBAL_CLASS_LABELS = list(range(TOTAL_CLASSES))


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_tensor(value):
    if torch.is_tensor(value):
        return value
    return torch.from_numpy(value)


def _load_features_pt_first(hdf5_file, pt_path: str, slide_id: str, coord_count: int):
    if os.path.exists(pt_path):
        pt_features = _torch_load_cpu(pt_path)
        if pt_features.shape[0] == coord_count:
            return pt_features

        print(
            "[WARN] PT/H5 patch-count mismatch; falling back to H5 features: "
            f"{slide_id} pt={tuple(pt_features.shape)} coords={coord_count}",
            file=sys.stderr,
            flush=True,
        )

    features = hdf5_file["features"][:]
    if features.shape[0] != coord_count:
        raise RuntimeError(
            "Feature/coord patch-count mismatch: "
            f"{slide_id} features={tuple(features.shape)} coords={coord_count}"
        )
    return features


def _patch_datasets() -> None:
    from mergeslide_tta import datasets as ds

    def generic_mil_getitem_pt_first(self, idx):
        slide_id = self.slide_data["slide_id"][idx]
        label = self.slide_data["label"][idx]
        stem = slide_id.split(".svs")[0]
        h5_path = os.path.join(self.data_dir, "h5_files", f"{stem}.h5")
        pt_path = os.path.join(self.data_dir, "pt_files", f"{stem}.pt")

        with h5py.File(h5_path, "r") as hdf5_file:
            coords = hdf5_file["coords"][:]
            features = _load_features_pt_first(
                hdf5_file, pt_path, slide_id, coords.shape[0]
            )

        return _as_tensor(features), torch.from_numpy(coords), label

    def generic_mil2_split_getitem_pt_first(self, idx):
        slide_id = self.data[idx]
        label = self.label[idx]
        h5_path = os.path.join(self.data_dir, "h5_files", f"{slide_id}.h5")
        pt_path = os.path.join(self.data_dir, "pt_files", f"{slide_id}.pt")

        with h5py.File(h5_path, "r") as hdf5_file:
            coords = hdf5_file["coords"][:]
            features = _load_features_pt_first(
                hdf5_file, pt_path, slide_id, coords.shape[0]
            )

        return _as_tensor(features), torch.from_numpy(coords), label

    ds.Generic_MIL_Dataset.__getitem__ = generic_mil_getitem_pt_first
    ds.Generic_MIL_Dataset2_Split.__getitem__ = generic_mil2_split_getitem_pt_first


def _patch_sklearn_per_class_metrics() -> None:
    import sklearn.metrics as sk_metrics

    original_recall_score = sk_metrics.recall_score
    original_precision_score = sk_metrics.precision_score

    def recall_score_fixed_labels(y_true, y_pred, **kwargs):
        if kwargs.get("average", "binary") is None:
            kwargs.setdefault("labels", GLOBAL_CLASS_LABELS)
            kwargs.setdefault("zero_division", 0)
        return original_recall_score(y_true, y_pred, **kwargs)

    def precision_score_fixed_labels(y_true, y_pred, **kwargs):
        if kwargs.get("average", "binary") is None:
            kwargs.setdefault("labels", GLOBAL_CLASS_LABELS)
            kwargs.setdefault("zero_division", 0)
        return original_precision_score(y_true, y_pred, **kwargs)

    sk_metrics.recall_score = recall_score_fixed_labels
    sk_metrics.precision_score = precision_score_fixed_labels


def _resolve_entrypoint(value: str) -> Path:
    raw_path = Path(value).expanduser()
    path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    path = path.resolve()

    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Entrypoint must be inside project root: {path}") from exc

    if not path.is_file():
        raise SystemExit(f"Entrypoint does not exist: {path}")
    return path


def _extract_entrypoint(argv: list[str]) -> tuple[Path, list[str]]:
    entrypoint = DEFAULT_ENTRYPOINT
    forwarded_args = []
    idx = 0

    env_entrypoint = os.environ.get("MERGESLIDE_EVAL_ENTRYPOINT")
    if env_entrypoint:
        entrypoint = _resolve_entrypoint(env_entrypoint)

    while idx < len(argv):
        arg = argv[idx]
        if arg == "--entrypoint":
            if idx + 1 >= len(argv):
                raise SystemExit("--entrypoint requires a path value")
            entrypoint = _resolve_entrypoint(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--entrypoint="):
            entrypoint = _resolve_entrypoint(arg.split("=", 1)[1])
            idx += 1
            continue

        forwarded_args.append(arg)
        idx += 1

    return entrypoint, forwarded_args


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    _patch_datasets()
    _patch_sklearn_per_class_metrics()
    entrypoint, forwarded_args = _extract_entrypoint(sys.argv[1:])

    print(
        "[INFO] PT-first feature loading and fixed-label per-class metrics enabled; "
        f"entrypoint={entrypoint.name}",
        file=sys.stderr,
        flush=True,
    )
    sys.argv = [str(entrypoint), *forwarded_args]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
