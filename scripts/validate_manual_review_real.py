"""Validate whether a manual review file looks like real review data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / "artifacts" / ".matplotlib").resolve()))

from xai_evidence_sod.xai.manual_review import (
    FAILURE_CAUSE_HYPOTHESIS_CHOICES,
    LABEL_QUALITY_CHOICES,
    RECOMMENDED_ACTION_CHOICES,
    SALIENCY_ALIGNMENT_CHOICES,
    VISUAL_EVIDENCE_QUALITY_CHOICES,
)

REQUIRED_REVIEW_COLUMNS = [
    "visual_evidence_quality",
    "saliency_alignment",
    "failure_cause_hypothesis",
    "label_quality",
    "recommended_action",
    "reviewer_notes",
]
DEMO_MARKERS = [
    "synthetic",
    "smoke-test",
    "demo annotation",
    "demo/synthetic",
    "exercise summary aggregation",
    "pipeline validation only",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate whether a manual review CSV is real review data.")
    parser.add_argument(
        "--manual-review",
        default="artifacts/manual_evidence_review_chipped/manual_review_filled.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/manual_evidence_review_chipped/manual_review_real_validation.json",
    )
    parser.add_argument(
        "--min-completed-rows",
        type=int,
        default=4,
        help="Minimum number of completed rows required for a passing validation.",
    )
    parser.add_argument(
        "--max-empty-share",
        type=float,
        default=0.25,
        help="Maximum allowed empty share for each required review column.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manual_review_path = (PROJECT_ROOT / args.manual_review).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    payload = validate_manual_review_real(
        manual_review_path=manual_review_path,
        min_completed_rows=args.min_completed_rows,
        max_empty_share=args.max_empty_share,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "passed" else 1)


def validate_manual_review_real(
    manual_review_path: Path,
    min_completed_rows: int,
    max_empty_share: float,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not manual_review_path.exists():
        errors.append(f"Manual review file does not exist: {manual_review_path}")
        return _build_payload(
            status="failed",
            manual_review_source="missing",
            manual_review_used_as_research_evidence=False,
            row_count=0,
            completed_row_count=0,
            errors=errors,
            warnings=warnings,
            column_empty_share={},
            invalid_values={},
        )

    rows = _read_csv(manual_review_path)
    if not rows:
        errors.append(f"Manual review file is empty: {manual_review_path}")
        return _build_payload(
            status="failed",
            manual_review_source="empty",
            manual_review_used_as_research_evidence=False,
            row_count=0,
            completed_row_count=0,
            errors=errors,
            warnings=warnings,
            column_empty_share={},
            invalid_values={},
        )

    missing_columns = [column for column in REQUIRED_REVIEW_COLUMNS if column not in rows[0]]
    if missing_columns:
        errors.append(f"Missing required review columns: {', '.join(missing_columns)}")

    file_text = manual_review_path.read_text(encoding="utf-8").lower()
    demo_hits = sorted(marker for marker in DEMO_MARKERS if marker in file_text)
    manual_review_source = "real_candidate"
    if demo_hits:
        manual_review_source = "demo_or_synthetic"
        errors.append(
            "Manual review still contains demo/synthetic markers: " + ", ".join(demo_hits)
        )

    completed_row_count = sum(1 for row in rows if _has_review_content(row))
    if completed_row_count < min_completed_rows:
        errors.append(
            f"Completed row count {completed_row_count} is below the minimum required {min_completed_rows}."
        )

    column_empty_share = _column_empty_share(rows, REQUIRED_REVIEW_COLUMNS)
    for column, empty_share in column_empty_share.items():
        if empty_share > max_empty_share:
            warnings.append(
                f"Column `{column}` has empty share {empty_share:.3f}, above the recommended limit {max_empty_share:.3f}."
            )

    invalid_values = _collect_invalid_values(rows)
    for column, values in invalid_values.items():
        if values:
            errors.append(f"Column `{column}` contains invalid values: {', '.join(values)}")

    if manual_review_source == "real_candidate" and completed_row_count > 0:
        partially_complete_rows = sum(1 for row in rows if _has_partial_review_content(row) and not _has_review_content(row))
        if partially_complete_rows:
            warnings.append(
                f"{partially_complete_rows} rows contain partial review content; finish them before treating the file as complete research evidence."
            )

    status = "passed" if not errors else "failed"
    manual_review_used_as_research_evidence = status == "passed"
    if status == "passed" and any(empty_share > max_empty_share for empty_share in column_empty_share.values()):
        warnings.append("Validation passed, but some required columns still have high empty share.")

    return _build_payload(
        status=status,
        manual_review_source=manual_review_source if status == "failed" or manual_review_source != "real_candidate" else "real_candidate",
        manual_review_used_as_research_evidence=manual_review_used_as_research_evidence,
        row_count=len(rows),
        completed_row_count=completed_row_count,
        errors=errors,
        warnings=warnings,
        column_empty_share=column_empty_share,
        invalid_values=invalid_values,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_payload(
    status: str,
    manual_review_source: str,
    manual_review_used_as_research_evidence: bool,
    row_count: int,
    completed_row_count: int,
    errors: list[str],
    warnings: list[str],
    column_empty_share: dict[str, float],
    invalid_values: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "status": status,
        "manual_review_source": manual_review_source,
        "manual_review_used_as_research_evidence": manual_review_used_as_research_evidence,
        "row_count": row_count,
        "completed_row_count": completed_row_count,
        "errors": errors,
        "warnings": warnings,
        "column_empty_share": {key: round(value, 6) for key, value in column_empty_share.items()},
        "invalid_values": invalid_values,
    }


def _has_review_content(row: dict[str, str]) -> bool:
    return all(str(row.get(column, "")).strip() for column in REQUIRED_REVIEW_COLUMNS)


def _has_partial_review_content(row: dict[str, str]) -> bool:
    return any(str(row.get(column, "")).strip() for column in REQUIRED_REVIEW_COLUMNS)


def _column_empty_share(rows: list[dict[str, str]], columns: list[str]) -> dict[str, float]:
    shares: dict[str, float] = {}
    total = len(rows)
    for column in columns:
        empty_count = sum(1 for row in rows if not str(row.get(column, "")).strip())
        shares[column] = empty_count / total if total else 1.0
    return shares


def _collect_invalid_values(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    allowed = {
        "visual_evidence_quality": set(VISUAL_EVIDENCE_QUALITY_CHOICES),
        "saliency_alignment": set(SALIENCY_ALIGNMENT_CHOICES),
        "failure_cause_hypothesis": set(FAILURE_CAUSE_HYPOTHESIS_CHOICES),
        "label_quality": set(LABEL_QUALITY_CHOICES),
        "recommended_action": set(RECOMMENDED_ACTION_CHOICES),
    }
    invalid: dict[str, list[str]] = {}
    for column, allowed_values in allowed.items():
        counter: Counter[str] = Counter()
        for row in rows:
            value = str(row.get(column, "")).strip()
            if value and value not in allowed_values:
                counter[value] += 1
        invalid[column] = sorted(counter.keys())
    return invalid


if __name__ == "__main__":
    main()
