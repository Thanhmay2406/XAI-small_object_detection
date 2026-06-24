"""Create a synthetic Phase 7 manual-review CSV for smoke testing only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VISUAL_EVIDENCE_QUALITY_VALUES = [
    "clear",
    "weak",
    "ambiguous",
    "not_visible",
    "clear",
    "weak",
]
SALIENCY_ALIGNMENT_VALUES = [
    "aligned",
    "partially_aligned",
    "misaligned",
    "absent",
    "uncertain",
    "aligned",
]
FAILURE_CAUSE_HYPOTHESIS_VALUES = [
    "small_object",
    "weak_visual_cue",
    "background_confusion",
    "localization_shift",
    "cam_method_uncertain",
    "possible_label_noise",
]
LABEL_QUALITY_VALUES = [
    "ok",
    "ok",
    "questionable",
    "ok",
    "uncertain",
    "likely_noise",
]
RECOMMENDED_ACTION_VALUES = [
    "chipped_focused_augmentation",
    "hard_sample_weighting",
    "background_negative_mining",
    "cross_method_xai_check",
    "no_action",
    "label_review",
]
REVIEWER_NOTE_VALUES = [
    "Synthetic smoke-test row 1; demo annotation only, not a research conclusion.",
    "Synthetic smoke-test row 2; values chosen only to exercise summary aggregation.",
    "Synthetic smoke-test row 3; reviewer note intentionally descriptive and non-causal.",
    "Synthetic smoke-test row 4; demo row for localization-shift style branch coverage.",
    "Synthetic smoke-test row 5; uncertainty is simulated for pipeline validation only.",
    "Synthetic smoke-test row 6; label-quality value is synthetic and not dataset QA evidence.",
]
REQUIRED_COLUMNS = [
    "visual_evidence_quality",
    "saliency_alignment",
    "failure_cause_hypothesis",
    "label_quality",
    "recommended_action",
    "reviewer_notes",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a synthetic manual_review_filled.csv for Phase 7 smoke testing."
    )
    parser.add_argument(
        "--template",
        default="artifacts/manual_evidence_review_chipped/manual_review_template.csv",
        help="Source manual review template CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/manual_evidence_review_chipped/manual_review_filled.csv",
        help="Destination synthetic filled CSV.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="How many top rows to copy from the template. Must be between 5 and 8.",
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_fieldnames(fieldnames: list[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Template CSV is missing required manual-review columns: {missing_text}")


def build_smoke_rows(template_rows: list[dict[str, str]], requested_rows: int) -> list[dict[str, str]]:
    if requested_rows < 5 or requested_rows > 8:
        raise ValueError("--rows must be between 5 and 8 for the requested smoke-test scope.")
    if len(template_rows) < requested_rows:
        raise ValueError(
            f"Template has only {len(template_rows)} rows, fewer than requested smoke-test size {requested_rows}."
        )

    outputs: list[dict[str, str]] = []
    for idx, row in enumerate(template_rows[:requested_rows]):
        filled = dict(row)
        filled["visual_evidence_quality"] = VISUAL_EVIDENCE_QUALITY_VALUES[idx % len(VISUAL_EVIDENCE_QUALITY_VALUES)]
        filled["saliency_alignment"] = SALIENCY_ALIGNMENT_VALUES[idx % len(SALIENCY_ALIGNMENT_VALUES)]
        filled["failure_cause_hypothesis"] = FAILURE_CAUSE_HYPOTHESIS_VALUES[
            idx % len(FAILURE_CAUSE_HYPOTHESIS_VALUES)
        ]
        filled["label_quality"] = LABEL_QUALITY_VALUES[idx % len(LABEL_QUALITY_VALUES)]
        filled["recommended_action"] = RECOMMENDED_ACTION_VALUES[idx % len(RECOMMENDED_ACTION_VALUES)]
        filled["reviewer_notes"] = REVIEWER_NOTE_VALUES[idx % len(REVIEWER_NOTE_VALUES)]
        outputs.append(filled)
    return outputs


def main() -> None:
    args = build_parser().parse_args()
    template_path = (PROJECT_ROOT / args.template).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    template_rows = _read_csv(template_path)
    if not template_rows:
        raise ValueError(f"Template CSV is empty: {template_path}")

    fieldnames = list(template_rows[0].keys())
    _validate_fieldnames(fieldnames)
    smoke_rows = build_smoke_rows(template_rows, args.rows)
    _write_csv(output_path, smoke_rows, fieldnames)

    print(f"Wrote {len(smoke_rows)} synthetic smoke-test rows to {output_path}")
    print("This file is demo/synthetic manual review content only, not a real research annotation result.")


if __name__ == "__main__":
    main()
