"""Phase placeholder for model layer inspection."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect detector layer names and indices.")
    parser.add_argument("--config", default="configs/train/baseline_yolo.yaml")
    parser.add_argument("--weights", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise NotImplementedError("Phase 4 will implement model layer inspection.")


if __name__ == "__main__":
    main()
