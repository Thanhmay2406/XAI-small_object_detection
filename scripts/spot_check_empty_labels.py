"""Spot-check empty-label images from a YOLO-format dataset."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw

from xai_evidence_sod.data import load_yolo_dataset_config
from xai_evidence_sod.utils.config import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample and export empty-label images for manual review.")
    parser.add_argument("--data", default="configs/dataset/drill_bit_yolo.yaml")
    parser.add_argument("--output", default="artifacts/dataset_spotcheck_empty")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-images", action="store_true", help="Copy sampled images instead of only listing them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yolo_dataset_config(args.data)
    output_dir = ensure_dir(args.output, "spot-check output directory", create=True)
    images_dir = ensure_dir(output_dir / "images", "spot-check image directory", create=True)

    empty_rows = _collect_empty_label_rows(config)
    if not empty_rows:
        raise RuntimeError("No empty-label images were found for the configured dataset.")

    rng = random.Random(args.seed)
    sample_count = min(args.num_samples, len(empty_rows))
    sampled_rows = rng.sample(empty_rows, sample_count)
    sampled_rows.sort(key=lambda row: (row["split"], row["image_name"]))

    csv_path = output_dir / "selected_empty_labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image_path", "label_path", "image_name"])
        writer.writeheader()
        writer.writerows(sampled_rows)

    if args.copy_images:
        for row in sampled_rows:
            source = Path(row["image_path"])
            destination = images_dir / f"{row['split']}__{source.name}"
            shutil.copy2(source, destination)

    contact_sheet_path = output_dir / "empty_label_contact_sheet.jpg"
    _build_contact_sheet(sampled_rows, contact_sheet_path)

    summary_path = output_dir / "README.md"
    summary_lines = [
        "# Empty Label Spot Check",
        "",
        f"- Dataset config: `{Path(args.data).resolve()}`",
        f"- Sample count: {sample_count}",
        f"- Seed: {args.seed}",
        f"- CSV list: `{csv_path.resolve()}`",
        f"- Contact sheet: `{contact_sheet_path.resolve()}`",
        "",
        "Manual review checklist:",
        "- Confirm whether sampled images are true negative images without visible defects.",
        "- Look for missed annotations in visually obvious defect regions.",
        "- Note whether empty labels cluster by split, scene condition, or capture pattern.",
        "- If many sampled images contain visible defects, revisit label quality before full baseline training.",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def _collect_empty_label_rows(config) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in config.splits:
        image_map = {path.stem: path for path in split.image_files}
        for label_path in split.label_files:
            if label_path.read_text(encoding="utf-8").strip():
                continue
            image_path = image_map.get(label_path.stem)
            if image_path is None:
                continue
            rows.append(
                {
                    "split": split.name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "image_name": image_path.name,
                }
            )
    return rows


def _build_contact_sheet(rows: list[dict[str, str]], output_path: Path, thumb_size: tuple[int, int] = (160, 160), columns: int = 4) -> None:
    tiles = []
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        image.thumbnail(thumb_size)
        tile = Image.new("RGB", thumb_size, color=(255, 255, 255))
        x_offset = (thumb_size[0] - image.width) // 2
        y_offset = (thumb_size[1] - image.height) // 2
        tile.paste(image, (x_offset, y_offset))
        tile = ImageOps.expand(tile, border=2, fill=(30, 30, 30))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile.width - 1, 20), fill=(30, 30, 30))
        draw.text((6, 4), f"{row['split']}:{Path(row['image_name']).stem[:18]}", fill=(255, 255, 255))
        tiles.append(tile)

    rows_needed = (len(tiles) + columns - 1) // columns
    sheet_width = columns * (thumb_size[0] + 4)
    sheet_height = rows_needed * (thumb_size[1] + 4)
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(245, 245, 245))

    for index, tile in enumerate(tiles):
        col = index % columns
        row = index // columns
        sheet.paste(tile, (col * (thumb_size[0] + 4), row * (thumb_size[1] + 4)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


if __name__ == "__main__":
    main()
