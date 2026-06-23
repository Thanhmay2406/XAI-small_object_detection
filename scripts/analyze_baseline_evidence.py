"""Phase placeholder for baseline evidence analysis."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze baseline evidence across feature levels.")
    parser.add_argument("--config", default="configs/train/baseline_yolo.yaml")
    parser.add_argument("--weights", default="")
    parser.add_argument("--out", default="artifacts/evidence_analysis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise NotImplementedError("Phase 7 will implement baseline evidence analysis.")


if __name__ == "__main__":
    main()
