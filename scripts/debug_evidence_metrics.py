"""Phase placeholder for evidence metric debugging."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug evidence metrics on sample inputs.")
    parser.add_argument("--config", default="configs/xai_evidence/default.yaml")
    parser.add_argument("--out", default="artifacts/evidence_debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise NotImplementedError("Phase 6 will implement evidence metric debugging.")


if __name__ == "__main__":
    main()
