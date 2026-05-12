#!/usr/bin/env python
"""Find and remove corrupt TRIDENT intermediate outputs for TCGA-RCC.

Default behavior is a dry run. Add --apply to delete files.
The script never deletes source WSI files.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/datastore/uittogether2/LuuTru/Thanhld/WSI/dataset/TCGA-RCC")
DEFAULT_PATCH_DIR = "10x_256px_0px_overlap"


def geojson_has_bad_tail(path: Path) -> tuple[bool, str]:
    """Fast check for GeoJSON files that were truncated during writing."""
    if not path.exists():
        return True, "missing"
    if path.stat().st_size == 0:
        return True, "empty"

    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 4096))
        tail = handle.read().strip()

    if not tail.endswith(b"}"):
        return True, "tail_does_not_end_with_closing_brace"
    return False, ""


def geojson_load_error(path: Path) -> str:
    """Validate syntax without requiring geopandas."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except Exception as exc:  # noqa: BLE001 - we want the exact validation failure.
        return f"{type(exc).__name__}: {exc}"
    return ""


def find_bad_geojsons(preprocessed_dir: Path, *, full_json_check: bool) -> list[dict[str, str]]:
    geojson_dir = preprocessed_dir / "contours_geojson"
    bad: list[dict[str, str]] = []
    for path in sorted(geojson_dir.glob("*.geojson")):
        bad_tail, reason = geojson_has_bad_tail(path)
        error = ""

        if bad_tail:
            error = geojson_load_error(path)
        elif full_json_check:
            error = geojson_load_error(path)
            if error:
                reason = "json_parse_error"

        if bad_tail or error:
            bad.append(
                {
                    "slide": path.stem,
                    "geojson": str(path),
                    "geojson_size": str(path.stat().st_size if path.exists() else -1),
                    "reason": reason or "json_parse_error",
                    "error": error,
                }
            )
    return bad


def paths_for_slide(preprocessed_dir: Path, patch_dir: str, slide: str) -> list[Path]:
    paths = [
        preprocessed_dir / "contours_geojson" / f"{slide}.geojson",
        preprocessed_dir / "contours" / f"{slide}.jpg",
        preprocessed_dir / "thumbnails" / f"{slide}.jpg",
        preprocessed_dir / patch_dir / "patches" / f"{slide}_patches.h5",
        preprocessed_dir / patch_dir / "patches" / f"{slide}_patches.h5.lock",
        preprocessed_dir / patch_dir / "visualization" / f"{slide}.jpg",
        preprocessed_dir / patch_dir / "features_conch_v15" / f"{slide}.h5",
        preprocessed_dir / patch_dir / "features_conch_v15" / f"{slide}.h5.lock",
    ]
    paths.extend(sorted((preprocessed_dir / "wsi_states").glob(f"{slide}*.json")))
    return paths


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slide",
        "problem_geojson",
        "problem_reason",
        "problem_error",
        "path",
        "exists_before",
        "size_before",
        "action",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean corrupt TRIDENT outputs for TCGA-RCC")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--patch-dir", default=DEFAULT_PATCH_DIR)
    parser.add_argument("--slide", action="append", help="Clean only this slide stem. Can be repeated.")
    parser.add_argument("--full-json-check", action="store_true", help="Parse every GeoJSON, slower but stricter.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Without this, dry-run only.")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    preprocessed_dir = dataset_root / "preprocessed"
    if not preprocessed_dir.is_dir():
        raise NotADirectoryError(preprocessed_dir)

    if args.slide:
        bad_geojsons = [
            {
                "slide": slide,
                "geojson": str(preprocessed_dir / "contours_geojson" / f"{slide}.geojson"),
                "geojson_size": "",
                "reason": "manual_slide",
                "error": "",
            }
            for slide in args.slide
        ]
    else:
        bad_geojsons = find_bad_geojsons(preprocessed_dir, full_json_check=args.full_json_check)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = args.manifest
    if manifest is None:
        manifest = Path(__file__).resolve().parent / "log" / f"RCC_corrupt_trident_cleanup_{timestamp}.csv"

    rows: list[dict[str, str]] = []
    for item in bad_geojsons:
        slide = item["slide"]
        for path in paths_for_slide(preprocessed_dir, args.patch_dir, slide):
            exists_before = path.exists()
            size_before = str(path.stat().st_size) if exists_before and path.is_file() else ""
            action = "missing"
            error = ""
            if exists_before:
                if args.apply:
                    try:
                        path.unlink()
                        action = "deleted"
                    except Exception as exc:  # noqa: BLE001 - report permission issues clearly.
                        action = "delete_failed"
                        error = f"{type(exc).__name__}: {exc}"
                else:
                    action = "would_delete"

            rows.append(
                {
                    "slide": slide,
                    "problem_geojson": item["geojson"],
                    "problem_reason": item["reason"],
                    "problem_error": item["error"],
                    "path": str(path),
                    "exists_before": str(exists_before),
                    "size_before": size_before,
                    "action": action,
                    "error": error,
                }
            )

    write_manifest(manifest, rows)

    print(f"dataset_root={dataset_root}")
    print(f"preprocessed_dir={preprocessed_dir}")
    print(f"bad_slides={len(bad_geojsons)}")
    for item in bad_geojsons:
        print(f"bad_slide={item['slide']} reason={item['reason']} size={item.get('geojson_size', '')}")
    print(f"apply={args.apply}")
    print(f"manifest={manifest}")
    print(f"would_delete={sum(1 for row in rows if row['action'] == 'would_delete')}")
    print(f"deleted={sum(1 for row in rows if row['action'] == 'deleted')}")
    print(f"delete_failed={sum(1 for row in rows if row['action'] == 'delete_failed')}")
    print(f"missing={sum(1 for row in rows if row['action'] == 'missing')}")


if __name__ == "__main__":
    main()
