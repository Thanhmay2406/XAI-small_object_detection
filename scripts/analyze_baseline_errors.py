"""Analyze baseline evaluation artifacts into research-oriented error summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xai_evidence_sod.evaluation.error_analysis import run_baseline_error_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze baseline evaluation errors without retraining the model.")
    parser.add_argument("--eval-dir", default="artifacts/baseline_eval")
    parser.add_argument("--data", default="configs/dataset/drill_bit_yolo.yaml")
    parser.add_argument("--output", default="artifacts/baseline_error_analysis")
    parser.add_argument("--focus-class", default="Chipped")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--low-conf-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_baseline_error_analysis(
        eval_dir=args.eval_dir,
        data_path=args.data,
        output_dir=args.output,
        focus_class=args.focus_class,
        imgsz=args.imgsz,
        iou_threshold=args.iou_threshold,
        low_conf_threshold=args.low_conf_threshold,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
