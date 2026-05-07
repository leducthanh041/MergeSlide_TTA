#!/usr/bin/env python
"""Convert TRIDENT patch feature H5 files to PyTorch PT tensors.

This script reads TRIDENT feature files containing a `features` dataset and
writes one tensor-only `.pt` file per slide. It does not modify or delete the
source `.h5` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None


DEFAULT_FEATURES_DIR = (
    "/datastore/uittogether2/LuuTru/Thanhld/WSI/dataset/TCGA-ESCA/"
    "preprocessed/10x_256px_0px_overlap/features_conch_v15"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TRIDENT feature .h5 files to tensor-only .pt files."
    )
    parser.add_argument(
        "--features-dir",
        default=DEFAULT_FEATURES_DIR,
        help="Directory containing TRIDENT feature .h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write .pt files. Defaults to a sibling directory named "
            "<features-dir-name>_pt."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing .pt files.",
    )
    return parser.parse_args()


def convert_one(h5_path: Path, pt_path: Path) -> tuple[bool, str]:
    try:
        with h5py.File(h5_path, "r") as handle:
            if "features" not in handle:
                return False, "missing `features` dataset"
            features = handle["features"][:]

        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(features), pt_path)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - keep batch conversion running.
        return False, str(exc)


def main() -> None:
    args = parse_args()

    features_dir = Path(args.features_dir).resolve()
    if not features_dir.is_dir():
        raise NotADirectoryError(features_dir)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else features_dir.with_name(f"{features_dir.name}_pt")
    )

    h5_files = sorted(features_dir.glob("*.h5"))
    converted = 0
    skipped = 0
    failed = 0

    print(f"features_dir={features_dir}")
    print(f"output_dir={output_dir}")
    print(f"h5_files={len(h5_files)}")
    print(f"dry_run={args.dry_run}")

    if args.dry_run:
        existing = sum(1 for path in h5_files if (output_dir / f"{path.stem}.pt").exists())
        print(f"would_convert={len(h5_files) - existing}")
        print(f"existing_pt={existing}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    iterator = h5_files
    if tqdm is not None:
        iterator = tqdm(h5_files, desc="Converting H5 to PT", unit="file")

    for h5_path in iterator:
        pt_path = output_dir / f"{h5_path.stem}.pt"
        if pt_path.exists() and not args.overwrite:
            skipped += 1
            continue

        ok, reason = convert_one(h5_path, pt_path)
        if ok:
            converted += 1
        else:
            failed += 1
            print(f"FAILED\t{h5_path}\t{reason}")

    print(f"converted={converted}")
    print(f"skipped_existing={skipped}")
    print(f"failed={failed}")
    print(f"pt_files={len(list(output_dir.glob('*.pt')))}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
