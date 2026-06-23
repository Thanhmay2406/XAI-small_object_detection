"""Phase placeholder for evidence-loss training."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train detector with evidence preservation loss.")
    parser.add_argument("--config", default="configs/train/evidence_loss_yolo.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise NotImplementedError("Phase 9 will implement evidence-loss training.")


if __name__ == "__main__":
    main()
