#!/usr/bin/env python3
"""
Visualize TCGA-CESC patch extraction results by patch_size.

Outputs:
  - summary.csv: metadata from patch H5 files
  - gallery.html: thumbnail gallery for all slides, grouped/color-coded by patch_size
  - comparison_256_vs_512.jpg: one representative 256 slide vs one representative 512 slide

The script only reads existing PrePATH outputs. It does not require raw .svs files.
"""

from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CESC_ROOT = SCRIPT_DIR / "PrePATH" / "downloaded_data" / "TCGA-CESC"
DEFAULT_PATCH_ROOT = DEFAULT_CESC_ROOT / "CESC_patches"
DEFAULT_OUT_DIR = DEFAULT_CESC_ROOT / "patch_size_visualization"


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    h5_path: Path
    stitch_path: Path | None
    mask_path: Path | None
    patch_size: int
    patch_level: int
    downsample_x: float
    downsample_y: float
    num_patches: int
    level_dim: str

    @property
    def patch_power_hint(self) -> str:
        return f"level={self.patch_level}, downsample=({self.downsample_x:.3f},{self.downsample_y:.3f})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a visual gallery and representative comparison for CESC patch sizes."
    )
    parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_ROOT / "patches")
    parser.add_argument("--stitch-dir", type=Path, default=DEFAULT_PATCH_ROOT / "stitches")
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_PATCH_ROOT / "masks")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--thumb-size", type=int, default=320)
    parser.add_argument(
        "--representative",
        choices=["max_patches", "median_patches", "first"],
        default="max_patches",
        help="How to choose one slide per patch_size for comparison.",
    )
    parser.add_argument(
        "--patch-sizes",
        nargs=2,
        type=int,
        default=[256, 512],
        help="Two patch sizes to compare.",
    )
    return parser.parse_args()


def resolve_image(slide_id: str, directory: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = directory / f"{slide_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_slide_record(h5_path: Path, stitch_dir: Path, mask_dir: Path) -> SlideRecord:
    slide_id = h5_path.stem
    with h5py.File(h5_path, "r") as h5_file:
        coords = h5_file["coords"]
        attrs = coords.attrs
        downsample = attrs.get("downsample", (-1.0, -1.0))
        level_dim = attrs.get("level_dim", "")

        if hasattr(downsample, "tolist"):
            downsample = downsample.tolist()
        if hasattr(level_dim, "tolist"):
            level_dim = level_dim.tolist()

        downsample_x = float(downsample[0]) if len(downsample) > 0 else -1.0
        downsample_y = float(downsample[1]) if len(downsample) > 1 else downsample_x

        return SlideRecord(
            slide_id=slide_id,
            h5_path=h5_path,
            stitch_path=resolve_image(slide_id, stitch_dir),
            mask_path=resolve_image(slide_id, mask_dir),
            patch_size=int(attrs.get("patch_size", -1)),
            patch_level=int(attrs.get("patch_level", -1)),
            downsample_x=downsample_x,
            downsample_y=downsample_y,
            num_patches=int(len(coords)),
            level_dim=str(level_dim),
        )


def load_records(patch_dir: Path, stitch_dir: Path, mask_dir: Path) -> list[SlideRecord]:
    h5_paths = sorted(patch_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No H5 patch files found in: {patch_dir}")
    return [read_slide_record(path, stitch_dir, mask_dir) for path in h5_paths]


def write_summary(records: Iterable[SlideRecord], output_path: Path) -> None:
    fields = [
        "slide_id",
        "patch_size",
        "patch_level",
        "downsample_x",
        "downsample_y",
        "num_patches",
        "level_dim",
        "h5_path",
        "stitch_path",
        "mask_path",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "slide_id": rec.slide_id,
                    "patch_size": rec.patch_size,
                    "patch_level": rec.patch_level,
                    "downsample_x": f"{rec.downsample_x:.6f}",
                    "downsample_y": f"{rec.downsample_y:.6f}",
                    "num_patches": rec.num_patches,
                    "level_dim": rec.level_dim,
                    "h5_path": rec.h5_path,
                    "stitch_path": rec.stitch_path or "",
                    "mask_path": rec.mask_path or "",
                }
            )


def make_thumbnail(src: Path, dst: Path, max_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (max_size, max_size), "white")
        x = (max_size - img.width) // 2
        y = (max_size - img.height) // 2
        canvas.paste(img, (x, y))
        canvas.save(dst, quality=90)


def create_thumbnails(records: Iterable[SlideRecord], thumb_dir: Path, max_size: int) -> dict[str, str]:
    thumb_paths: dict[str, str] = {}
    for rec in records:
        src = rec.stitch_path or rec.mask_path
        if src is None:
            continue
        dst = thumb_dir / f"{rec.slide_id}.jpg"
        make_thumbnail(src, dst, max_size)
        thumb_paths[rec.slide_id] = dst.name
    return thumb_paths


def patch_size_class(patch_size: int) -> str:
    if patch_size == 256:
        return "ps256"
    if patch_size == 512:
        return "ps512"
    return "psother"


def write_gallery(records: list[SlideRecord], thumb_paths: dict[str, str], out_dir: Path) -> None:
    counts: dict[int, int] = {}
    for rec in records:
        counts[rec.patch_size] = counts.get(rec.patch_size, 0) + 1

    cards = []
    for rec in sorted(records, key=lambda r: (r.patch_size, -r.num_patches, r.slide_id)):
        thumb = thumb_paths.get(rec.slide_id)
        if thumb is None:
            img_tag = "<div class='missing'>No stitch/mask image</div>"
        else:
            img_tag = f"<img src='thumbs/{html.escape(thumb)}' loading='lazy' alt='{html.escape(rec.slide_id)}'>"
        stitch_link = html.escape(str(rec.stitch_path)) if rec.stitch_path else ""
        mask_link = html.escape(str(rec.mask_path)) if rec.mask_path else ""
        cards.append(
            f"""
            <article class="card {patch_size_class(rec.patch_size)}">
              {img_tag}
              <div class="meta">
                <strong>{html.escape(rec.slide_id)}</strong>
                <span>patch_size={rec.patch_size}, patches={rec.num_patches}</span>
                <span>{html.escape(rec.patch_power_hint)}</span>
                <a href="{stitch_link}">stitch</a>
                <a href="{mask_link}">mask</a>
              </div>
            </article>
            """
        )

    count_text = " | ".join(f"patch_size {k}: {v}" for k, v in sorted(counts.items()))
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TCGA-CESC Patch Size Gallery</title>
  <style>
    :root {{
      --ink: #171717;
      --paper: #f4efe7;
      --card: #fffaf0;
      --blue: #164e63;
      --orange: #9a3412;
      --gray: #525252;
    }}
    body {{
      margin: 0;
      padding: 28px;
      background: radial-gradient(circle at top left, #fff7d6, transparent 32rem), var(--paper);
      color: var(--ink);
      font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
    }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    .summary {{ margin: 0 0 24px; color: var(--gray); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--card);
      border: 4px solid var(--gray);
      box-shadow: 6px 6px 0 rgba(0, 0, 0, 0.14);
      padding: 10px;
    }}
    .card.ps256 {{ border-color: var(--blue); }}
    .card.ps512 {{ border-color: var(--orange); }}
    .card img {{
      width: 100%;
      aspect-ratio: 1;
      object-fit: contain;
      background: white;
      display: block;
    }}
    .meta {{
      display: grid;
      gap: 4px;
      margin-top: 10px;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .meta strong {{ font-size: 14px; }}
    .meta a {{ color: var(--ink); margin-right: 8px; }}
    .missing {{
      height: 220px;
      display: grid;
      place-items: center;
      background: white;
      color: var(--gray);
    }}
  </style>
</head>
<body>
  <h1>TCGA-CESC Patch Size Gallery</h1>
  <p class="summary">{html.escape(count_text)}</p>
  <section class="grid">
    {''.join(cards)}
  </section>
</body>
</html>
"""
    (out_dir / "gallery.html").write_text(page)


def choose_representative(records: list[SlideRecord], patch_size: int, mode: str) -> SlideRecord:
    candidates = [rec for rec in records if rec.patch_size == patch_size and (rec.stitch_path or rec.mask_path)]
    if not candidates:
        raise ValueError(f"No visualizable slide found for patch_size={patch_size}")
    candidates = sorted(candidates, key=lambda r: (r.num_patches, r.slide_id))
    if mode == "first":
        return sorted(candidates, key=lambda r: r.slide_id)[0]
    if mode == "median_patches":
        return candidates[len(candidates) // 2]
    return candidates[-1]


def draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str]) -> None:
    font = ImageFont.load_default()
    x, y = xy
    line_h = 16
    width = max(draw.textlength(line, font=font) for line in lines) + 20
    height = line_h * len(lines) + 14
    draw.rectangle((x, y, x + int(width), y + height), fill=(255, 250, 240), outline=(20, 20, 20), width=2)
    for idx, line in enumerate(lines):
        draw.text((x + 10, y + 8 + idx * line_h), line, fill=(20, 20, 20), font=font)


def render_slide_panel(rec: SlideRecord, panel_size: tuple[int, int]) -> Image.Image:
    src = rec.stitch_path or rec.mask_path
    if src is None:
        raise ValueError(f"No stitch/mask image for {rec.slide_id}")

    panel_w, panel_h = panel_size
    header_h = 92
    image_area = (panel_w, panel_h - header_h)

    panel = Image.new("RGB", panel_size, "white")
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail(image_area, Image.Resampling.LANCZOS)
        x = (panel_w - img.width) // 2
        y = header_h + (image_area[1] - img.height) // 2
        panel.paste(img, (x, y))

    draw = ImageDraw.Draw(panel)
    lines = [
        f"patch_size={rec.patch_size}",
        f"num_patches={rec.num_patches}",
        f"patch_level={rec.patch_level}",
        f"downsample=({rec.downsample_x:.3f},{rec.downsample_y:.3f})",
        rec.slide_id[:96],
    ]
    draw_text_box(draw, (12, 12), lines)
    return panel


def write_comparison(records: list[SlideRecord], patch_sizes: list[int], mode: str, out_path: Path) -> tuple[SlideRecord, SlideRecord]:
    left = choose_representative(records, patch_sizes[0], mode)
    right = choose_representative(records, patch_sizes[1], mode)

    panel_size = (900, 900)
    gutter = 28
    canvas = Image.new("RGB", (panel_size[0] * 2 + gutter, panel_size[1]), (244, 239, 231))
    canvas.paste(render_slide_panel(left, panel_size), (0, 0))
    canvas.paste(render_slide_panel(right, panel_size), (panel_size[0] + gutter, 0))
    canvas.save(out_path, quality=92)
    return left, right


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.patch_dir, args.stitch_dir, args.mask_dir)
    write_summary(records, args.out_dir / "summary.csv")
    thumb_paths = create_thumbnails(records, args.out_dir / "thumbs", args.thumb_size)
    write_gallery(records, thumb_paths, args.out_dir)
    left, right = write_comparison(
        records,
        args.patch_sizes,
        args.representative,
        args.out_dir / f"comparison_{args.patch_sizes[0]}_vs_{args.patch_sizes[1]}.jpg",
    )

    counts: dict[int, int] = {}
    for rec in records:
        counts[rec.patch_size] = counts.get(rec.patch_size, 0) + 1

    print("Output directory:", args.out_dir)
    print("Patch size counts:", dict(sorted(counts.items())))
    print("Gallery:", args.out_dir / "gallery.html")
    print("Summary:", args.out_dir / "summary.csv")
    print("Comparison:", args.out_dir / f"comparison_{args.patch_sizes[0]}_vs_{args.patch_sizes[1]}.jpg")
    print("Selected left:", left.slide_id, left.patch_size, left.num_patches)
    print("Selected right:", right.slide_id, right.patch_size, right.num_patches)


if __name__ == "__main__":
    main()
