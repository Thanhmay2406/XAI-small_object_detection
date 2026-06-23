"""Sample prediction or error-case images for manual review."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from xai_evidence_sod.utils.config import ensure_dir, ensure_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy or list sampled error-case images from a CSV artifact.")
    parser.add_argument("--csv", default="artifacts/baseline_eval_smoke/chipped_error_cases.csv")
    parser.add_argument("--output", default="artifacts/baseline_error_samples")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-images", action="store_true", help="Copy sampled images into the output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path = ensure_file(args.csv, "Error-case CSV")
    output_dir = ensure_dir(args.output, "Error sample output directory", create=True)
    images_dir = ensure_dir(output_dir / "images", "Error sample image directory", create=True)

    rows = _load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in error-case CSV: {csv_path}")

    rng = random.Random(args.seed)
    sample_count = min(args.num_samples, len(rows))
    sampled_rows = rng.sample(rows, sample_count)
    sampled_rows.sort(key=lambda row: row["image_path"])

    sampled_csv = output_dir / "sampled_error_cases.csv"
    with sampled_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sampled_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sampled_rows)

    if args.copy_images:
        for row in sampled_rows:
            image_path = Path(row["image_path"]).expanduser().resolve()
            destination = images_dir / image_path.name
            shutil.copy2(image_path, destination)

    readme_path = output_dir / "README.md"
    readme_lines = [
        "# Error Case Samples",
        "",
        f"- Source CSV: `{csv_path.resolve()}`",
        f"- Sample count: {sample_count}",
        f"- Seed: {args.seed}",
        f"- Sampled CSV: `{sampled_csv.resolve()}`",
        "",
        "Manual review checklist:",
        "- Check whether the target object is visible and localized enough for the detector to catch.",
        "- For `Chipped`, look for low-contrast or weak-edge defects that may align with the weak-evidence framing.",
        "- Compare false-negative images against normal-looking negative images from the empty-label spot check.",
    ]
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
