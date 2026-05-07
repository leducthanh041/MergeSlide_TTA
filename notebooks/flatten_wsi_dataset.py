#!/usr/bin/env python
"""Flatten WSI files from nested folders into a dataset root.

The script moves only files matching the requested extension. It writes a CSV
manifest before applying changes so the old and new locations are auditable.
After moving, original source folders can optionally be removed even if they
still contain GDC sidecar files such as logs/*.parcel or annotations.txt.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path


def collect_files(root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for current_root, _, filenames in os.walk(root):
        current = Path(current_root)
        for filename in filenames:
            path = current / filename
            if path.suffix.lower() in extensions:
                files.append(path)
    return sorted(files)


def source_dirs_from_rows(rows: list[dict[str, str]]) -> list[Path]:
    dirs = {
        Path(row["source"]).parent
        for row in rows
        if row.get("status") in {"planned", "moved"} and row.get("source")
    }
    return sorted(dirs)


def load_source_dirs_from_manifest(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return source_dirs_from_rows(rows)


def remove_source_dirs(source_dirs: list[Path], root: Path, extensions: set[str]) -> tuple[int, int]:
    removed = 0
    skipped = 0
    for directory in sorted(set(source_dirs), key=lambda item: len(item.parts), reverse=True):
        if directory == root or not directory.exists():
            skipped += 1
            continue

        remaining_wsi = [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ]
        if remaining_wsi:
            skipped += 1
            continue

        shutil.rmtree(directory)
        removed += 1
    return removed, skipped


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "source", "destination", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def build_plan(root: Path, files: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    nested = [path for path in files if path.parent != root]
    root_files = [path for path in files if path.parent == root]

    nested_name_counts = Counter(path.name for path in nested)
    root_names = {path.name for path in root_files}

    plan: list[dict[str, str]] = []
    errors: list[str] = []
    for source in nested:
        destination = root / source.name
        reason = ""
        status = "planned"

        if nested_name_counts[source.name] > 1:
            status = "collision"
            reason = "duplicate filename among nested WSI files"
            errors.append(f"duplicate nested filename: {source.name}")
        elif source.name in root_names:
            status = "collision"
            reason = "destination already exists at dataset root"
            errors.append(f"destination exists: {destination}")

        plan.append(
            {
                "status": status,
                "source": str(source),
                "destination": str(destination),
                "reason": reason,
            }
        )

    return plan, sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Move nested WSI files into a dataset root")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--ext", default=".svs", help="File extension to move, e.g. .svs")
    parser.add_argument("--manifest", required=True, help="CSV manifest path")
    parser.add_argument("--apply", action="store_true", help="Actually move files")
    parser.add_argument(
        "--remove-empty-dirs",
        "--remove-source-dirs",
        dest="remove_source_dirs",
        action="store_true",
        help=(
            "Remove original source folders after moving. This deletes sidecar "
            "files in those folders, but skips any folder that still contains "
            "a file matching --ext."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    extensions = {args.ext.lower() if args.ext.startswith(".") else f".{args.ext.lower()}"}
    files = collect_files(root, extensions)
    root_count = sum(1 for path in files if path.parent == root)
    nested_count = len(files) - root_count

    manifest_path = Path(args.manifest)
    plan, errors = build_plan(root, files)
    source_dirs_to_remove = source_dirs_from_rows(plan)

    if not plan and args.apply and args.remove_source_dirs:
        # Support cleanup after a previous successful move.
        source_dirs_to_remove = load_source_dirs_from_manifest(manifest_path)
    else:
        write_manifest(manifest_path, plan)

    print(f"root={root}")
    print(f"extension={','.join(sorted(extensions))}")
    print(f"root_files={root_count}")
    print(f"nested_files={nested_count}")
    print(f"planned_moves={sum(1 for row in plan if row['status'] == 'planned')}")
    print(f"collisions={sum(1 for row in plan if row['status'] == 'collision')}")
    print(f"manifest={args.manifest}")

    if errors:
        print("Refusing to move because collisions were found:")
        for error in errors[:50]:
            print(f"  {error}")
        raise SystemExit(1)

    if not args.apply:
        print("dry_run=true")
        return

    moved = 0
    for row in plan:
        source = Path(row["source"])
        destination = Path(row["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        row["status"] = "moved"
        moved += 1

    write_manifest(Path(args.manifest), plan)

    removed_dirs = 0
    skipped_remove_dirs = 0
    if args.remove_source_dirs:
        removed_dirs, skipped_remove_dirs = remove_source_dirs(source_dirs_to_remove, root, extensions)

    final_root_count = len([path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions])
    final_nested_count = len([path for path in collect_files(root, extensions) if path.parent != root])

    print(f"moved={moved}")
    print(f"removed_source_dirs={removed_dirs}")
    print(f"skipped_remove_source_dirs={skipped_remove_dirs}")
    print(f"final_root_files={final_root_count}")
    print(f"final_nested_files={final_nested_count}")


if __name__ == "__main__":
    main()
