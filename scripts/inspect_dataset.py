"""Inspect a YOLO-format dataset and export reproducible diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xai_evidence_sod.data import inspect_dataset, load_yolo_dataset_config, write_dataset_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a YOLO-format dataset.")
    parser.add_argument("--data", default="configs/dataset/drill_bit_yolo.yaml", help="Path to a YOLO dataset YAML file.")
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        default="artifacts/dataset_inspection",
        help="Directory where inspection artifacts will be written.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Projected square image size used for pixel-space bbox statistics.")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_yolo_dataset_config(args.data)
    inspection = inspect_dataset(config, imgsz=args.imgsz, max_samples=args.max_samples)
    output_paths = write_dataset_outputs(inspection, args.output)

    payload = {
        "dataset_config_path": str(config.config_path),
        "dataset_root": str(config.dataset_root),
        "imgsz": args.imgsz,
        "outputs": output_paths,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
