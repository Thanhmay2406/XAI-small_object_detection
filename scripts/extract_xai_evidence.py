"""Phase 5 post-hoc XAI evidence extraction on curated baseline cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xai_evidence_sod.xai import run_xai_evidence_extraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract post-hoc XAI evidence from curated baseline cases.")
    parser.add_argument("--weights", default="experiments/baseline_drill_bit/weights/best.pt")
    parser.add_argument("--data", default="configs/dataset/drill_bit_yolo.yaml")
    parser.add_argument("--cases", default="artifacts/baseline_error_analysis/focus_class_error_cases.csv")
    parser.add_argument("--prediction-rows", default="artifacts/baseline_eval/prediction_rows.csv")
    parser.add_argument("--output", default="artifacts/xai_evidence_chipped")
    parser.add_argument("--focus-class", default="Chipped")
    parser.add_argument("--methods", default="eigencam", help="Comma-separated list. Phase 5 implements eigencam first.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-layer", type=int, default=18)
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    outputs = run_xai_evidence_extraction(
        weights_path=args.weights,
        data_path=args.data,
        cases_csv=args.cases,
        output_dir=args.output,
        focus_class=args.focus_class,
        methods=methods,
        max_cases=args.max_cases,
        seed=args.seed,
        imgsz=args.imgsz,
        device=args.device,
        target_layer=args.target_layer,
        prediction_rows_csv=args.prediction_rows,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
