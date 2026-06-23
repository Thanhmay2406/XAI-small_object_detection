"""Phase placeholder for run comparison."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline and evidence-guided runs.")
    parser.add_argument("--config", default="configs/xai_evidence/default.yaml")
    parser.add_argument("--out", default="artifacts/comparisons")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise NotImplementedError("Phase 10 will implement run comparison.")


if __name__ == "__main__":
    main()
