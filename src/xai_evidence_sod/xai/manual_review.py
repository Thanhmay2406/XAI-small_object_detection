"""Phase 7 manual evidence review preparation and summarization."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VISUAL_EVIDENCE_QUALITY_CHOICES = ["clear", "weak", "ambiguous", "not_visible"]
SALIENCY_ALIGNMENT_CHOICES = ["aligned", "partially_aligned", "misaligned", "absent", "uncertain"]
FAILURE_CAUSE_HYPOTHESIS_CHOICES = [
    "weak_visual_cue",
    "small_object",
    "background_confusion",
    "localization_shift",
    "class_confusion",
    "possible_label_noise",
    "cam_method_uncertain",
    "other",
]
LABEL_QUALITY_CHOICES = ["ok", "questionable", "likely_noise", "uncertain"]
RECOMMENDED_ACTION_CHOICES = [
    "hard_sample_weighting",
    "chipped_focused_augmentation",
    "background_negative_mining",
    "label_review",
    "cross_method_xai_check",
    "no_action",
]
MANUAL_TEMPLATE_COLUMNS = [
    "case_id",
    "image_path",
    "case_type",
    "bucket",
    "error_type",
    "tags",
    "energy_in_gt_box",
    "energy_in_pred_box",
    "peak_inside_gt_box",
    "saliency_concentration",
    "overlay_path",
    "crop_path",
    "visual_evidence_quality",
    "saliency_alignment",
    "failure_cause_hypothesis",
    "label_quality",
    "recommended_action",
    "reviewer_notes",
]


def prepare_manual_evidence_review(
    representatives_csv: str | Path,
    group_summary_csv: str | Path,
    output_dir: str | Path,
    focus_class: str,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Create Phase 7 manual review artifacts and optional summaries."""

    project_root_path = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    representatives_path = Path(representatives_csv)
    group_summary_path = Path(group_summary_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    representative_rows = _read_csv(representatives_path)
    group_summary_rows = _read_csv(group_summary_path)
    manual_template_rows = _build_manual_template_rows(representative_rows, project_root_path)

    outputs = {
        "manual_review_template_csv": _write_csv(manual_template_rows, output_path / "manual_review_template.csv"),
        "manual_review_guide_md": _write_guide(
            output_dir=output_path,
            focus_class=focus_class,
            group_summary_rows=group_summary_rows,
            representative_rows=representative_rows,
        ),
        "intervention_decision_table_md": _write_decision_table(output_path / "intervention_decision_table.md"),
        "readme_md": _write_readme(output_path, focus_class, representative_rows),
    }

    filled_csv_path = output_path / "manual_review_filled.csv"
    if filled_csv_path.exists():
        summary_rows, summary_json = summarize_manual_review(
            filled_csv=filled_csv_path,
            representatives_csv=representatives_path,
            focus_class=focus_class,
        )
        outputs["manual_review_summary_csv"] = _write_csv(summary_rows, output_path / "manual_review_summary.csv")
        outputs["manual_review_summary_json"] = _write_json(summary_json, output_path / "manual_review_summary.json")

    return outputs


def summarize_manual_review(
    filled_csv: str | Path,
    representatives_csv: str | Path,
    focus_class: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate a manually filled review CSV."""

    filled_rows = _read_csv(Path(filled_csv))
    representative_rows = _read_csv(Path(representatives_csv))
    if not filled_rows:
        raise ValueError(f"Manual review file is empty: {filled_csv}")

    merged_rows = _merge_rows_on_case_and_bucket(filled_rows, representative_rows)
    summary_rows: list[dict[str, Any]] = []
    for column in [
        "visual_evidence_quality",
        "saliency_alignment",
        "failure_cause_hypothesis",
        "label_quality",
        "recommended_action",
    ]:
        counter = Counter(row.get(column, "") for row in merged_rows if row.get(column))
        for value, count in sorted(counter.items()):
            summary_rows.append(
                {
                    "summary_type": column,
                    "summary_value": value,
                    "count": count,
                    "share": round(count / len(merged_rows), 6),
                }
            )

    bucket_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in merged_rows:
        bucket = row.get("bucket", "")
        action = row.get("recommended_action", "")
        if bucket and action:
            bucket_action_counts[bucket][action] += 1
    for bucket, counter in sorted(bucket_action_counts.items()):
        top_action, top_count = counter.most_common(1)[0]
        summary_rows.append(
            {
                "summary_type": "bucket_top_action",
                "summary_value": bucket,
                "count": top_count,
                "share": round(top_count / sum(counter.values()), 6),
                "top_action": top_action,
            }
        )

    summary_json = {
        "focus_class": focus_class,
        "reviewed_row_count": len(merged_rows),
        "completed_row_count": sum(1 for row in merged_rows if _has_review_content(row)),
        "column_counts": {
            column: dict(Counter(row.get(column, "") for row in merged_rows if row.get(column)))
            for column in [
                "visual_evidence_quality",
                "saliency_alignment",
                "failure_cause_hypothesis",
                "label_quality",
                "recommended_action",
            ]
        },
        "bucket_top_actions": {
            bucket: counter.most_common(1)[0][0]
            for bucket, counter in sorted(bucket_action_counts.items())
            if counter
        },
        "notes": [
            "Manual-review summaries are descriptive annotations, not causal proof.",
            "These outputs support Phase 8 intervention design review only; they do not train or validate any intervention.",
        ],
    }
    return summary_rows, summary_json


def _build_manual_template_rows(representative_rows: list[dict[str, str]], project_root: Path) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in representative_rows:
        case_id = row.get("case_id", "")
        bucket = row.get("representative_group", "")
        key = (case_id, bucket)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        case_group = row.get("case_group", "")
        outputs.append(
            {
                "case_id": case_id,
                "image_path": _normalize_relative(row.get("image_path"), project_root),
                "case_type": case_group,
                "bucket": bucket,
                "error_type": _infer_error_type(case_group, bucket),
                "tags": _build_tags(row),
                "energy_in_gt_box": row.get("energy_in_gt_box", ""),
                "energy_in_pred_box": row.get("energy_in_pred_box", ""),
                "peak_inside_gt_box": row.get("peak_inside_gt_box", ""),
                "saliency_concentration": row.get("saliency_concentration", ""),
                "overlay_path": _normalize_relative(row.get("overlay_path"), project_root),
                "crop_path": _normalize_relative(row.get("crop_path"), project_root),
                "visual_evidence_quality": "",
                "saliency_alignment": "",
                "failure_cause_hypothesis": "",
                "label_quality": "",
                "recommended_action": "",
                "reviewer_notes": "",
            }
        )
    return outputs


def _infer_error_type(case_group: str, bucket: str) -> str:
    if case_group == "false_negative":
        return "missed_detection"
    if case_group == "false_positive":
        return "background_or_spurious_detection"
    if case_group == "localization_error":
        return "localization_shift"
    if case_group == "true_positive_proxy":
        if "near_threshold" in bucket:
            return "borderline_true_positive"
        return "reference_true_positive"
    return "other"


def _build_tags(row: dict[str, str]) -> str:
    tags = [row.get("case_group", ""), row.get("representative_group", "")]
    if str(row.get("is_near_threshold", "")).lower() == "true":
        tags.append("near_threshold")
    if row.get("method"):
        tags.append(str(row["method"]))
    return "|".join(tag for tag in tags if tag)


def _merge_rows_on_case_and_bucket(
    filled_rows: list[dict[str, str]], representative_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    index = {(row.get("case_id", ""), row.get("representative_group", "")): row for row in representative_rows}
    merged: list[dict[str, str]] = []
    for row in filled_rows:
        bucket = row.get("bucket", "") or row.get("representative_group", "")
        base = index.get((row.get("case_id", ""), bucket), {})
        merged.append({**base, **row, "bucket": bucket})
    return merged


def _has_review_content(row: dict[str, str]) -> bool:
    for key in [
        "visual_evidence_quality",
        "saliency_alignment",
        "failure_cause_hypothesis",
        "label_quality",
        "recommended_action",
        "reviewer_notes",
    ]:
        if row.get(key):
            return True
    return False


def _write_guide(
    output_dir: Path,
    focus_class: str,
    group_summary_rows: list[dict[str, str]],
    representative_rows: list[dict[str, str]],
) -> Path:
    bucket_counts = Counter(row.get("representative_group", "") for row in representative_rows if row.get("representative_group"))
    lines = [
        "# Manual Evidence Review Guide",
        "",
        f"This guide standardizes manual review for Phase 7 `{focus_class}` evidence cases.",
        "",
        "## Review goal",
        "",
        "- Verify whether the selected Phase 6 representative cases show stable post-hoc evidence patterns.",
        "- Record descriptive judgments about evidence quality, alignment, and plausible failure mode hypotheses.",
        "- Prepare a decision gate for Phase 8 intervention design without training any model.",
        "",
        "## How to review each row",
        "",
        "1. Open the overlay and crop for the row in `manual_review_template.csv`.",
        "2. Check whether the highlighted region is visually meaningful for the labeled object or predicted region.",
        "3. Fill the categorical fields using the provided schema values only when possible.",
        "4. Use `reviewer_notes` for uncertainty, caveats, or observations that do not fit the controlled labels.",
        "5. Avoid causal claims. Record only descriptive visual patterns and intervention ideas to be tested later.",
        "",
        "## Review buckets in this run",
        "",
    ]
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- `{bucket}`: `{count}` rows")
    lines.extend(
        [
            "",
            "## Phase 6 quantitative reference",
            "",
        ]
    )
    for row in group_summary_rows:
        lines.append(
            "- `{group}`: count={count}, mean_gt={gt}, mean_pred={pred}, peak_rate={peak}, mean_saliency={sal}".format(
                group=row.get("group_name", ""),
                count=row.get("count", ""),
                gt=row.get("energy_in_gt_box_mean", ""),
                pred=row.get("energy_in_pred_box_mean", ""),
                peak=row.get("peak_inside_gt_box_rate", ""),
                sal=row.get("saliency_concentration_mean", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Allowed values",
            "",
            f"- `visual_evidence_quality`: {', '.join(VISUAL_EVIDENCE_QUALITY_CHOICES)}",
            f"- `saliency_alignment`: {', '.join(SALIENCY_ALIGNMENT_CHOICES)}",
            f"- `failure_cause_hypothesis`: {', '.join(FAILURE_CAUSE_HYPOTHESIS_CHOICES)}",
            f"- `label_quality`: {', '.join(LABEL_QUALITY_CHOICES)}",
            f"- `recommended_action`: {', '.join(RECOMMENDED_ACTION_CHOICES)}",
            "",
            "## Interpretation boundary",
            "",
            "- Manual review is used to prioritize intervention hypotheses, not to prove detector mechanisms.",
            "- Phase 7 does not train, validate, or compare any intervention.",
        ]
    )
    output_path = output_dir / "manual_review_guide.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_decision_table(output_path: Path) -> Path:
    lines = [
        "# Intervention Decision Table",
        "",
        "| Observed manual-review pattern | Suggested Phase 8 candidate | Gate condition | Notes |",
        "| --- | --- | --- | --- |",
        "| Repeated low GT evidence on false negatives | `hard_sample_weighting` | Seen across multiple `fn_low_evidence` rows with consistent manual agreement | Prioritize difficult `Chipped` misses only after review consensus |",
        "| Repeated evidence on very small or faint `Chipped` regions | `chipped_focused_augmentation` | Manual notes repeatedly cite weak cues or small-object visibility limits | Keep this as a training hypothesis only; do not claim it will help yet |",
        "| False positives repeatedly attend to background texture or artifact regions | `background_negative_mining` | `fp_background_like_evidence` rows repeatedly show background-linked saliency | Candidate should be tested only after label quality looks acceptable |",
        "| Multiple rows suggest questionable or noisy labels | `label_review` | `label_quality` frequently marked `questionable` or `likely_noise` | This gate should be satisfied before intervention training |",
        "| Manual review often cannot trust the EigenCAM pattern itself | `cross_method_xai_check` | Many rows marked `cam_method_uncertain` or `saliency_alignment=uncertain` | Add contrastive post-hoc XAI inspection before designing training changes |",
        "| Manual review does not support a stable recurring pattern | `no_action` | No intervention candidate has repeated support across buckets | Stay descriptive and avoid premature Phase 8 intervention design |",
        "",
        "## Phase 8 decision gate",
        "",
        "- Proceed only if a candidate pattern is repeated across several manually reviewed rows and the notes remain descriptive rather than causal.",
        "- Prefer `label_review` or `cross_method_xai_check` before any training intervention when label quality or CAM trustworthiness is uncertain.",
        "- Do not proceed from one-off visually striking cases alone.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_readme(output_dir: Path, focus_class: str, representative_rows: list[dict[str, str]]) -> Path:
    bucket_counts = Counter(row.get("representative_group", "") for row in representative_rows if row.get("representative_group"))
    lines = [
        "# Phase 7 Manual Evidence Review",
        "",
        f"This directory prepares manual review artifacts for the `{focus_class}` Phase 6 representative evidence cases.",
        "",
        "## Files",
        "",
        "- `manual_review_template.csv`",
        "- `manual_review_guide.md`",
        "- `intervention_decision_table.md`",
        "- `README.md`",
        "- optional `manual_review_summary.csv` and `manual_review_summary.json` if `manual_review_filled.csv` is present",
        "",
        "## Buckets",
        "",
    ]
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- `{bucket}`: `{count}` rows")
    lines.extend(
        [
            "",
            "## Reminder",
            "",
            "- Phase 7 does not train any intervention.",
            "- Phase 7 does not establish causal interpretation for EigenCAM patterns.",
            "- If `manual_review_filled.csv` was created by the smoke-test helper, any `manual_review_summary.*` outputs are demo/synthetic validation artifacts only and not real research findings.",
        ]
    )
    output_path = output_dir / "README.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _normalize_relative(value: str | None, project_root: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path)
    return value
