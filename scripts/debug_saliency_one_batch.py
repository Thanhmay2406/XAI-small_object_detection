"""Phase placeholder for saliency debugging."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate saliency for one batch.")
    parser.add_argument("--config", default="configs/train/baseline_yolo.yaml")
    parser.add_argument("--weights", default="")
    parser.add_argument("--out", default="artifacts/xai_debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise NotImplementedError("Phase 5 will implement saliency debugging.")


if __name__ == "__main__":
    main()
