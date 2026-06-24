"""Phase 6 descriptive review of post-hoc XAI evidence artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / "artifacts" / ".matplotlib").resolve()))

from xai_evidence_sod.xai import review_xai_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review post-hoc XAI evidence patterns by case group.")
    parser.add_argument("--evidence-csv", default="artifacts/xai_evidence_chipped/evidence_cases.csv")
    parser.add_argument("--output", default="artifacts/xai_evidence_review_chipped")
    parser.add_argument("--focus-class", default="Chipped")
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = review_xai_evidence(
        evidence_csv=args.evidence_csv,
        output_dir=args.output,
        focus_class=args.focus_class,
        top_k=args.top_k,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
