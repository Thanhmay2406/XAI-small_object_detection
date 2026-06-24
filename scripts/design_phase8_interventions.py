"""Phase 8 intervention decision design CLI."""

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

from xai_evidence_sod.xai.intervention_design import design_phase8_interventions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Design Phase 8 intervention candidates from existing evidence.")
    parser.add_argument("--representatives", default="artifacts/xai_evidence_review_chipped/representative_cases.csv")
    parser.add_argument("--group-summary", default="artifacts/xai_evidence_review_chipped/evidence_group_summary.csv")
    parser.add_argument(
        "--manual-review-summary",
        default="artifacts/manual_evidence_review_chipped/manual_review_summary.csv",
    )
    parser.add_argument(
        "--manual-review-filled",
        default="artifacts/manual_evidence_review_chipped/manual_review_filled.csv",
    )
    parser.add_argument("--output", default="artifacts/intervention_design_chipped")
    parser.add_argument("--focus-class", default="Chipped")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = design_phase8_interventions(
        representatives_csv=args.representatives,
        group_summary_csv=args.group_summary,
        manual_review_summary_csv=args.manual_review_summary,
        manual_review_filled_csv=args.manual_review_filled,
        output_dir=args.output,
        focus_class=args.focus_class,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
