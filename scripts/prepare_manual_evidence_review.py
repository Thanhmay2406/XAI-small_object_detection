"""Phase 7 manual evidence review preparation CLI."""

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

from xai_evidence_sod.xai.manual_review import prepare_manual_evidence_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Phase 7 manual evidence review artifacts.")
    parser.add_argument("--representatives", default="artifacts/xai_evidence_review_chipped/representative_cases.csv")
    parser.add_argument("--group-summary", default="artifacts/xai_evidence_review_chipped/evidence_group_summary.csv")
    parser.add_argument("--output", default="artifacts/manual_evidence_review_chipped")
    parser.add_argument("--focus-class", default="Chipped")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = prepare_manual_evidence_review(
        representatives_csv=args.representatives,
        group_summary_csv=args.group_summary,
        output_dir=args.output,
        focus_class=args.focus_class,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
