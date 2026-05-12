#!/usr/bin/env python
"""Zip extracted TCGA dataset folders, then optionally remove originals.

The script stores files without compression because extracted WSI artifacts
(.h5, .pt, .jpg) usually do not compress enough to justify the CPU cost.
Original folders are deleted only after the zip integrity check succeeds and
only when --delete-after-verify is provided.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/datastore/uittogether3/LuuTru/Thanhld/WSI/dataset")
DEFAULT_ZIP_NAME = "Trident_extracted.zip"
DEFAULT_DATASET_DIRS = (
    "TCGA-BRCA",
    "TCGA-CESC",
    "TCGA-ESCA",
    "TCGA-NSCLC",
    "TCGA-RCC",
    "TCGA-TGCT",
)


def iter_files(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file():
            yield path


def zip_folder(zip_handle: zipfile.ZipFile, folder: Path, root: Path) -> tuple[int, int]:
    file_count = 0
    byte_count = 0

    for file_path in iter_files(folder):
        arcname = file_path.relative_to(root).as_posix()
        zip_handle.write(file_path, arcname)
        file_count += 1
        byte_count += file_path.stat().st_size

        if file_count % 1000 == 0:
            gib = byte_count / 1024**3
            print(f"[ZIP] {folder.name}: files={file_count:,} size={gib:.2f} GiB", flush=True)

    return file_count, byte_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive extracted TCGA dataset folders into one zip file."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing TCGA-* dataset folders.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Output zip path. Default: <dataset-root>/TCGA_extracted_datasets_20260513.zip",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=None,
        help="Dataset folder name to include. Can be repeated. Default: all configured TCGA folders.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing zip file.")
    parser.add_argument(
        "--delete-after-verify",
        action="store_true",
        help="Delete original dataset folders after zip integrity check succeeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    zip_path = args.zip_path.resolve() if args.zip_path else dataset_root / DEFAULT_ZIP_NAME
    dataset_names = tuple(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASET_DIRS

    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    if zip_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{zip_path} already exists. Use --overwrite if needed.")
        zip_path.unlink()

    folders = [dataset_root / name for name in dataset_names]
    missing = [folder for folder in folders if not folder.is_dir()]
    folders = [folder for folder in folders if folder.is_dir()]

    if missing:
        print("[WARN] missing folders will be skipped:")
        for folder in missing:
            print(f"  {folder}")

    if not folders:
        raise RuntimeError("No dataset folders found to zip.")

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] zip_path={zip_path}")
    print("[INFO] folders:")
    for folder in folders:
        print(f"  {folder}")

    total_files = 0
    total_bytes = 0
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as handle:
        for folder in folders:
            file_count, byte_count = zip_folder(handle, folder, dataset_root)
            total_files += file_count
            total_bytes += byte_count
            print(
                f"[DONE] {folder.name}: files={file_count:,} size={byte_count / 1024**3:.2f} GiB",
                flush=True,
            )

    print(f"[INFO] zip_created={zip_path}")
    print(f"[INFO] total_files={total_files:,}")
    print(f"[INFO] total_input_size={total_bytes / 1024**3:.2f} GiB")

    print("[INFO] testing zip integrity...", flush=True)
    with zipfile.ZipFile(zip_path, mode="r") as handle:
        bad_file = handle.testzip()

    if bad_file is not None:
        raise RuntimeError(f"Zip integrity check failed at: {bad_file}")

    print("[INFO] zip integrity OK")

    if not args.delete_after_verify:
        print("[INFO] original folders were kept. Add --delete-after-verify to remove them.")
        return

    print("[INFO] deleting original folders...", flush=True)
    for folder in folders:
        print(f"[DELETE] {folder}", flush=True)
        shutil.rmtree(folder)
    print("[INFO] delete completed")


if __name__ == "__main__":
    main()
