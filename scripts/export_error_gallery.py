"""Export a sampled manual-review gallery from baseline error-analysis CSV rows."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample and render baseline error cases for manual review.")
    parser.add_argument("--csv", default="artifacts/baseline_error_analysis/focus_class_error_cases.csv")
    parser.add_argument("--output", default="artifacts/baseline_error_gallery")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Error-case CSV not found: {csv_path}")

    output_dir = Path(args.output).expanduser().resolve()
    preview_dir = output_dir / "previews"
    crop_dir = output_dir / "crops"
    preview_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in error-case CSV: {csv_path}")

    rng = random.Random(args.seed)
    sample_count = min(args.num_samples, len(rows))
    sampled_rows = rng.sample(rows, sample_count)
    sampled_rows.sort(key=lambda row: (row.get("primary_error_type", ""), row["image_path"]))

    export_rows: list[dict[str, str]] = []
    preview_paths: list[Path] = []
    for sample_index, row in enumerate(sampled_rows):
        image_path = Path(row["image_path"]).expanduser().resolve()
        if not image_path.exists():
            continue
        preview_path = preview_dir / f"{sample_index:03d}__{image_path.stem}.jpg"
        crop_path = crop_dir / f"{sample_index:03d}__{image_path.stem}.jpg"
        _render_preview(image_path, row, preview_path)
        _render_crop(image_path, row, crop_path)
        preview_paths.append(preview_path)
        export_row = dict(row)
        export_row["preview_path"] = str(preview_path)
        export_row["crop_path"] = str(crop_path)
        export_rows.append(export_row)

    sampled_csv = output_dir / "sampled_errors.csv"
    with sampled_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(export_rows[0].keys()) if export_rows else list(rows[0].keys()))
        writer.writeheader()
        if export_rows:
            writer.writerows(export_rows)

    contact_sheet_path = output_dir / "error_gallery_contact_sheet.jpg"
    if preview_paths:
        _build_contact_sheet(preview_paths, contact_sheet_path)

    readme_path = output_dir / "README.md"
    readme_lines = [
        "# Baseline Error Gallery",
        "",
        f"- Source CSV: `{csv_path}`",
        f"- Sample count requested: `{args.num_samples}`",
        f"- Sample count exported: `{len(export_rows)}`",
        f"- Seed: `{args.seed}`",
        f"- Sampled CSV: `{sampled_csv}`",
        f"- Contact sheet: `{contact_sheet_path}`",
        "",
        "## Manual Review Checklist",
        "",
        "- Start with rows tagged `localization_error` or `class_confusion` because they are more informative than pure misses for Phase 5 evidence hooks.",
        "- For `Chipped`, compare the GT box region against the predicted box to judge whether the defect is visibly weak, spatially ambiguous, or potentially mislabeled.",
        "- Review zero-IoU false negatives separately from near-threshold errors because they may reflect different evidence regimes.",
        "- If many rows tagged `ambiguous_or_possible_label_noise` look annotation-borderline, record that before drawing modeling conclusions.",
    ]
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _render_preview(image_path: Path, row: dict[str, str], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    gt_box = _parse_box(row, "gt")
    pred_box = _parse_box(row, "pred")
    if gt_box is not None:
        draw.rectangle(gt_box, outline=(0, 220, 80), width=4)
    if pred_box is not None:
        draw.rectangle(pred_box, outline=(255, 120, 0), width=4)

    banner_height = 32
    banner = Image.new("RGB", (image.width, banner_height), color=(24, 24, 24))
    merged = Image.new("RGB", (image.width, image.height + banner_height))
    merged.paste(banner, (0, 0))
    merged.paste(image, (0, banner_height))
    banner_draw = ImageDraw.Draw(merged)
    banner_text = f"{row.get('primary_error_type', 'error')} | pred={row.get('pred_class_name', '-') or '-'} | gt={row.get('gt_class_name', '-') or '-'}"
    banner_draw.text((8, 8), banner_text, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(output_path, quality=90)


def _render_crop(image_path: Path, row: dict[str, str], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    gt_box = _parse_box(row, "gt")
    pred_box = _parse_box(row, "pred")
    reference = gt_box or pred_box
    if reference is None:
        image.save(output_path, quality=90)
        return

    margin = 32
    x1 = max(0, int(reference[0]) - margin)
    y1 = max(0, int(reference[1]) - margin)
    x2 = min(image.width, int(reference[2]) + margin)
    y2 = min(image.height, int(reference[3]) + margin)
    crop = image.crop((x1, y1, x2, y2))
    draw = ImageDraw.Draw(crop)
    if gt_box is not None:
        draw.rectangle((gt_box[0] - x1, gt_box[1] - y1, gt_box[2] - x1, gt_box[3] - y1), outline=(0, 220, 80), width=3)
    if pred_box is not None:
        draw.rectangle((pred_box[0] - x1, pred_box[1] - y1, pred_box[2] - x1, pred_box[3] - y1), outline=(255, 120, 0), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=90)


def _parse_box(row: dict[str, str], prefix: str) -> tuple[float, float, float, float] | None:
    x1 = row.get(f"{prefix}_x1", "")
    y1 = row.get(f"{prefix}_y1", "")
    x2 = row.get(f"{prefix}_x2", "")
    y2 = row.get(f"{prefix}_y2", "")
    if "" in {x1, y1, x2, y2}:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def _build_contact_sheet(preview_paths: list[Path], output_path: Path, thumb_size: tuple[int, int] = (256, 256), columns: int = 4) -> None:
    tiles = []
    for path in preview_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size)
        tile = Image.new("RGB", thumb_size, color=(245, 245, 245))
        x_offset = (thumb_size[0] - image.width) // 2
        y_offset = (thumb_size[1] - image.height) // 2
        tile.paste(image, (x_offset, y_offset))
        tile = ImageOps.expand(tile, border=2, fill=(30, 30, 30))
        tiles.append(tile)

    rows_needed = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (thumb_size[0] + 4), rows_needed * (thumb_size[1] + 4)), color=(250, 250, 250))
    for index, tile in enumerate(tiles):
        col = index % columns
        row = index // columns
        sheet.paste(tile, (col * (thumb_size[0] + 4), row * (thumb_size[1] + 4)))
    sheet.save(output_path, quality=90)


if __name__ == "__main__":
    main()
