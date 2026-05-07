#!/usr/bin/env python
"""Create a TRIDENT custom WSI CSV and skip slides without MPP metadata."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import openslide


MPP_COMMENT_RE = re.compile(r"(?:^|\|)\s*MPP\s*=\s*([0-9.]+)", re.IGNORECASE)


def parse_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mpp_from_properties(props: dict[str, str]) -> tuple[float | None, str]:
    for key in [
        openslide.PROPERTY_NAME_MPP_X,
        "openslide.mpp-x",
        "aperio.MPP",
        "openslide.mirax.MPP",
    ]:
        value = parse_float(props.get(key))
        if value is not None:
            return round(value, 4), key

    for key in ["openslide.comment", "tiff.ImageDescription"]:
        text = props.get(key, "")
        match = MPP_COMMENT_RE.search(text)
        if match:
            value = parse_float(match.group(1))
            if value is not None:
                return round(value, 4), f"{key}:MPP"

    x_resolution = parse_float(props.get("tiff.XResolution"))
    unit = str(props.get("tiff.ResolutionUnit", ""))
    if x_resolution:
        if unit.lower() == "centimeter":
            return round(10000.0 / x_resolution, 4), "tiff.XResolution/cm"
        if unit.upper() == "INCH":
            return round(25400.0 / x_resolution, 4), "tiff.XResolution/inch"

    return None, "missing_mpp"


def collect_wsi_paths(root: Path, extensions: set[str], search_nested: bool) -> list[Path]:
    if search_nested:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()
    return sorted(path for path in candidates if path.is_file() and path.suffix.lower() in extensions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TRIDENT custom_list_of_wsis CSV from MPP metadata")
    parser.add_argument("--wsi_dir", required=True, help="Directory containing WSIs")
    parser.add_argument("--output_csv", required=True, help="CSV for TRIDENT --custom_list_of_wsis")
    parser.add_argument("--skipped_csv", required=True, help="CSV listing slides skipped due to missing/unreadable MPP")
    parser.add_argument("--wsi_ext", nargs="+", default=[".svs"], help="WSI extensions to scan")
    parser.add_argument("--search_nested", action="store_true", help="Search nested directories")
    parser.add_argument("--progress_every", type=int, default=100, help="Print progress every N slides")
    args = parser.parse_args()

    wsi_dir = Path(args.wsi_dir).resolve()
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.wsi_ext}
    paths = collect_wsi_paths(wsi_dir, extensions, args.search_nested)

    output_csv = Path(args.output_csv)
    skipped_csv = Path(args.skipped_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    skipped_csv.parent.mkdir(parents=True, exist_ok=True)

    kept_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []

    for index, path in enumerate(paths, start=1):
        rel_path = os.path.relpath(path, wsi_dir)
        try:
            slide = openslide.OpenSlide(str(path))
            props = dict(slide.properties)
            mpp, source = mpp_from_properties(props)
            slide.close()
        except Exception as exc:
            skipped_rows.append({"wsi": rel_path, "reason": f"open_error: {exc}"})
            continue

        if mpp is None:
            skipped_rows.append({"wsi": rel_path, "reason": source})
        else:
            kept_rows.append({"wsi": rel_path, "mpp": f"{mpp:g}", "mpp_source": source})

        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"processed={index}/{len(paths)} kept={len(kept_rows)} skipped={len(skipped_rows)}", flush=True)

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wsi", "mpp", "mpp_source"])
        writer.writeheader()
        writer.writerows(kept_rows)

    with skipped_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wsi", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"total={len(paths)}")
    print(f"kept_with_mpp={len(kept_rows)}")
    print(f"skipped_missing_mpp={len(skipped_rows)}")
    print(f"output_csv={output_csv}")
    print(f"skipped_csv={skipped_csv}")


if __name__ == "__main__":
    main()
