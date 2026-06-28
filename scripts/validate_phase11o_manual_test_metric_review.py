"""Validate Phase 11O manual test metric review without executing evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11O"
DEFAULT_PHASE11N_SUMMARY = (
    "artifacts/phase11n_test_evaluation_output_collection_and_validation/"
    "phase11n_test_evaluation_output_summary.json"
)
DEFAULT_OUTPUT_DIR = "artifacts/phase11o_manual_test_metric_review"
TEMPLATE_CSV_NAME = "phase11o_manual_metric_review_template.csv"
USED_CSV_NAME = "phase11o_manual_metric_review_used.csv"
VALIDATION_CSV_NAME = "phase11o_manual_metric_validation.csv"
SUMMARY_JSON_NAME = "phase11o_manual_metric_review_summary.json"
MANIFEST_JSON_NAME = "phase11o_non_execution_manifest.json"
README_NAME = "README.md"
STATUS_PENDING = "phase11o_manual_test_metric_review_pending"
STATUS_REVIEW_PENDING = "phase11o_manual_test_metrics_validated_pending_manual_review_decision"
STATUS_REJECTED = "phase11o_manual_test_metrics_validated_rejected_needs_rerun_or_better_logs"
STATUS_VALIDATED_BLOCKED = "phase11o_manual_test_metrics_validated_reporting_still_blocked_missing_allow_flag"
STATUS_VALIDATED_ALLOWED = "phase11o_manual_test_metrics_validated_reporting_allowed_with_caveat"
NEXT_FILL = "fill_phase11o_manual_metric_review_csv_from_kaggle_visible_output_then_rerun_phase11o"
NEXT_REVIEW_DECISION = "finalize_phase11o_manual_metric_reviewer_decision_then_rerun_phase11o"
NEXT_REJECTED = "rerun_phase11m1_or_collect_better_manual_metric_logs_in_a_separate_approved_phase"
NEXT_ALLOW_FLAG = "rerun_phase11o_with_allow_reporting_with_caveat_or_keep_metrics_review_only"
NEXT_REPORT = "phase11p_prepare_final_test_evaluation_report_with_provenance_caveat"
ALLOWED_PHASES = {"11O", "Phase 11O"}
ALLOWED_SOURCE_TYPES = {
    "kaggle_notebook_visible_output",
    "saved_manual_note",
    "copied_ultralytics_console_summary",
    "other_manual_source",
}
ALLOWED_REVIEWER_DECISIONS = {
    "approved_for_reporting_with_caveat",
    "rejected_needs_rerun_or_better_logs",
    "pending_manual_review",
}
TEMPLATE_FIELDS = [
    "phase",
    "metric_source_type",
    "metric_source_description",
    "metric_source_path_or_note",
    "metric_extracted_by",
    "metric_extracted_at_local",
    "test_precision",
    "test_recall",
    "test_map50",
    "test_map50_95",
    "source_confirms_test_split",
    "source_confirms_model_best_pt",
    "source_confirms_dataset_yaml",
    "reviewer_decision",
    "reviewer_notes",
]
VALIDATION_FIELDS = ["check_name", "passed", "severity", "observed_value", "expected_value", "notes"]
REQUIRED_PHASE11N_KEYS = {
    "phase",
    "status",
    "eval_dir",
    "training_dir",
    "phase11l_best_epoch",
    "phase11l_best_metric_map50_95",
    "phase11l_final_epoch",
    "phase11l_final_metric_map50_95",
    "phase11l_final_metric_map50",
    "predictions_json_available",
    "evaluation_outputs_credible",
    "needs_manual_metric_review",
    "provenance_caveat_carried_forward",
}
BOOLEAN_FIELDS = [
    "source_confirms_test_split",
    "source_confirms_model_best_pt",
    "source_confirms_dataset_yaml",
]
METRIC_FIELDS = [
    "test_precision",
    "test_recall",
    "test_map50",
    "test_map50_95",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a manually extracted Phase 11M.1 test metric review without executing evaluation."
    )
    parser.add_argument("--phase11n-summary", default=DEFAULT_PHASE11N_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manual-review-csv", default="")
    parser.add_argument("--allow-reporting-with-caveat", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_phase11o_manual_test_metric_review(
        phase11n_summary_path=resolve_repo_path(args.phase11n_summary),
        output_dir=resolve_repo_path(args.output_dir),
        manual_review_csv_path=resolve_repo_path(args.manual_review_csv) if args.manual_review_csv else None,
        allow_reporting_with_caveat=args.allow_reporting_with_caveat,
    )
    concise = {
        "status": summary["status"],
        "manual_metrics_available": summary["manual_metrics_available"],
        "test_metrics_validated": summary["test_metrics_validated"],
        "reporting_allowed": summary["reporting_allowed"],
        "next_allowed_step": summary["next_allowed_step"],
    }
    print(json.dumps(concise, indent=2))


def validate_phase11o_manual_test_metric_review(
    phase11n_summary_path: Path,
    output_dir: Path,
    manual_review_csv_path: Path | None,
    allow_reporting_with_caveat: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    phase11n_summary = read_json_required(phase11n_summary_path, "Phase 11N summary")
    validation_rows: list[dict[str, Any]] = []
    validate_phase11n_summary(phase11n_summary, validation_rows, phase11n_summary_path)

    template_row = build_template_row(phase11n_summary)
    write_csv(output_dir / TEMPLATE_CSV_NAME, [template_row], TEMPLATE_FIELDS)

    review_result = validate_optional_manual_review_csv(
        manual_review_csv_path=manual_review_csv_path,
        template_row=template_row,
        validation_rows=validation_rows,
    )

    used_rows = review_result["used_rows"] if review_result["used_rows"] else [template_row]
    write_csv(output_dir / USED_CSV_NAME, used_rows, TEMPLATE_FIELDS)

    summary = build_summary(
        phase11n_summary=phase11n_summary,
        phase11n_summary_path=phase11n_summary_path,
        output_dir=output_dir,
        manual_review_csv_path=manual_review_csv_path,
        allow_reporting_with_caveat=allow_reporting_with_caveat,
        review_result=review_result,
    )
    manifest = build_non_execution_manifest(summary=summary, manual_review_csv_path=manual_review_csv_path)
    write_csv(output_dir / VALIDATION_CSV_NAME, validation_rows, VALIDATION_FIELDS)
    write_json(output_dir / SUMMARY_JSON_NAME, summary)
    write_json(output_dir / MANIFEST_JSON_NAME, manifest)
    (output_dir / README_NAME).write_text(build_readme(summary, manual_review_csv_path), encoding="utf-8")
    return summary


def build_summary(
    phase11n_summary: dict[str, Any],
    phase11n_summary_path: Path,
    output_dir: Path,
    manual_review_csv_path: Path | None,
    allow_reporting_with_caveat: bool,
    review_result: dict[str, Any],
) -> dict[str, Any]:
    valid = review_result["valid"]
    normalized_row = review_result["normalized_row"]
    reviewer_decision = normalized_row.get("reviewer_decision", "") if normalized_row else ""

    manual_metrics_available = valid
    test_metrics_validated = valid
    reporting_allowed = False
    status = STATUS_PENDING
    next_allowed_step = NEXT_FILL

    if valid and reviewer_decision == "approved_for_reporting_with_caveat":
        if allow_reporting_with_caveat:
            status = STATUS_VALIDATED_ALLOWED
            reporting_allowed = True
            next_allowed_step = NEXT_REPORT
        else:
            status = STATUS_VALIDATED_BLOCKED
            next_allowed_step = NEXT_ALLOW_FLAG
    elif valid and reviewer_decision == "pending_manual_review":
        status = STATUS_REVIEW_PENDING
        next_allowed_step = NEXT_REVIEW_DECISION
    elif valid and reviewer_decision == "rejected_needs_rerun_or_better_logs":
        status = STATUS_REJECTED
        next_allowed_step = NEXT_REJECTED

    return {
        "phase": PHASE,
        "status": status,
        "phase11n_summary_path": str(phase11n_summary_path),
        "phase11n_status": phase11n_summary["status"],
        "phase11n_eval_dir": phase11n_summary["eval_dir"],
        "phase11n_training_dir": phase11n_summary["training_dir"],
        "phase11n_predictions_json_available": phase11n_summary["predictions_json_available"],
        "phase11n_evaluation_outputs_credible": phase11n_summary["evaluation_outputs_credible"],
        "phase11n_needs_manual_metric_review": phase11n_summary["needs_manual_metric_review"],
        "phase11l_best_epoch": phase11n_summary["phase11l_best_epoch"],
        "phase11l_best_metric_map50_95": phase11n_summary["phase11l_best_metric_map50_95"],
        "phase11l_final_epoch": phase11n_summary["phase11l_final_epoch"],
        "phase11l_final_metric_map50_95": phase11n_summary["phase11l_final_metric_map50_95"],
        "phase11l_final_metric_map50": phase11n_summary["phase11l_final_metric_map50"],
        "provenance_caveat_carried_forward": phase11n_summary["provenance_caveat_carried_forward"],
        "manual_review_csv_provided": manual_review_csv_path is not None,
        "manual_review_csv_path": str(manual_review_csv_path) if manual_review_csv_path else "",
        "manual_review_csv_exists": manual_review_csv_path.exists() if manual_review_csv_path else False,
        "manual_metrics_available": manual_metrics_available,
        "test_metrics_validated": test_metrics_validated,
        "reporting_allowed": reporting_allowed,
        "allow_reporting_with_caveat_flag_used": allow_reporting_with_caveat,
        "reviewer_decision": reviewer_decision,
        "metric_source_type": normalized_row.get("metric_source_type", "") if normalized_row else "",
        "metric_source_description": normalized_row.get("metric_source_description", "") if normalized_row else "",
        "metric_source_path_or_note": normalized_row.get("metric_source_path_or_note", "") if normalized_row else "",
        "metric_extracted_by": normalized_row.get("metric_extracted_by", "") if normalized_row else "",
        "metric_extracted_at_local": normalized_row.get("metric_extracted_at_local", "") if normalized_row else "",
        "test_precision": maybe_float(normalized_row.get("test_precision")) if normalized_row else None,
        "test_recall": maybe_float(normalized_row.get("test_recall")) if normalized_row else None,
        "test_map50": maybe_float(normalized_row.get("test_map50")) if normalized_row else None,
        "test_map50_95": maybe_float(normalized_row.get("test_map50_95")) if normalized_row else None,
        "source_confirms_test_split": maybe_bool(normalized_row.get("source_confirms_test_split")) if normalized_row else None,
        "source_confirms_model_best_pt": maybe_bool(normalized_row.get("source_confirms_model_best_pt")) if normalized_row else None,
        "source_confirms_dataset_yaml": maybe_bool(normalized_row.get("source_confirms_dataset_yaml")) if normalized_row else None,
        "validation_error_count": len(review_result["errors"]),
        "validation_errors": review_result["errors"],
        "artifacts": {
            "template_csv": str(output_dir / TEMPLATE_CSV_NAME),
            "used_csv": str(output_dir / USED_CSV_NAME),
            "validation_csv": str(output_dir / VALIDATION_CSV_NAME),
            "summary_json": str(output_dir / SUMMARY_JSON_NAME),
            "non_execution_manifest_json": str(output_dir / MANIFEST_JSON_NAME),
            "readme_md": str(output_dir / README_NAME),
        },
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_phase11n_summary(
    phase11n_summary: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    phase11n_summary_path: Path,
) -> None:
    missing = sorted(REQUIRED_PHASE11N_KEYS.difference(phase11n_summary))
    if missing:
        raise SystemExit(
            f"Phase 11N summary is missing required keys at {phase11n_summary_path}: {', '.join(missing)}"
        )

    add_check(
        validation_rows,
        "phase11n_summary_exists",
        True,
        "info",
        str(phase11n_summary_path),
        "existing Phase 11N summary JSON",
        "Phase 11O requires the strict non-execution Phase 11N summary as input.",
    )
    add_check(
        validation_rows,
        "phase11n_phase_matches",
        phase11n_summary.get("phase") == "11N",
        "error",
        phase11n_summary.get("phase"),
        "11N",
        "Phase 11O should only consume a Phase 11N summary.",
    )
    add_check(
        validation_rows,
        "phase11n_status_needs_manual_metric_review",
        phase11n_summary.get("status") == "phase11n_test_evaluation_outputs_collected_needs_manual_metric_review",
        "error",
        phase11n_summary.get("status"),
        "phase11n_test_evaluation_outputs_collected_needs_manual_metric_review",
        "Phase 11O is the follow-up gate for manual metric review after Phase 11N.",
    )
    add_check(
        validation_rows,
        "phase11n_eval_outputs_credible",
        phase11n_summary.get("evaluation_outputs_credible") is True,
        "error",
        phase11n_summary.get("evaluation_outputs_credible"),
        True,
        "Phase 11O assumes a credible evaluation directory already exists.",
    )
    add_check(
        validation_rows,
        "phase11n_manual_review_required",
        phase11n_summary.get("needs_manual_metric_review") is True,
        "error",
        phase11n_summary.get("needs_manual_metric_review"),
        True,
        "Phase 11O should only be used when Phase 11N could not parse aggregate metrics automatically.",
    )


def validate_optional_manual_review_csv(
    manual_review_csv_path: Path | None,
    template_row: dict[str, str],
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "errors": [],
        "normalized_row": None,
        "used_rows": [],
    }
    if manual_review_csv_path is None:
        add_check(
            validation_rows,
            "manual_review_csv_provided",
            False,
            "info",
            "",
            "path to a filled manual review CSV",
            "No manual review CSV was provided, so Phase 11O stays pending and emits a template.",
        )
        return result

    exists = manual_review_csv_path.exists()
    add_check(
        validation_rows,
        "manual_review_csv_exists",
        exists,
        "error",
        str(manual_review_csv_path),
        "existing CSV path",
        "A provided manual review CSV must exist.",
    )
    if not exists:
        result["errors"].append(f"Manual review CSV not found: {manual_review_csv_path}")
        return result

    try:
        rows = read_csv_rows(manual_review_csv_path)
    except Exception as exc:  # pragma: no cover - surfaced in validation output
        result["errors"].append(f"Failed to read manual review CSV: {exc}")
        add_check(
            validation_rows,
            "manual_review_csv_readable",
            False,
            "error",
            type(exc).__name__,
            "readable CSV",
            "CSV parsing failed.",
        )
        return result

    add_check(
        validation_rows,
        "manual_review_csv_readable",
        True,
        "info",
        str(manual_review_csv_path),
        "readable CSV",
        "CSV was parsed successfully.",
    )
    add_check(
        validation_rows,
        "manual_review_csv_has_required_columns",
        bool(rows) and set(TEMPLATE_FIELDS).issubset(rows[0].keys()),
        "error",
        sorted(list(rows[0].keys())) if rows else [],
        TEMPLATE_FIELDS,
        "Manual review CSV must contain the exact required columns.",
    )
    add_check(
        validation_rows,
        "manual_review_csv_row_count",
        len(rows) == 1,
        "error",
        len(rows),
        1,
        "Phase 11O accepts exactly one primary review row.",
    )
    result["used_rows"] = [normalize_row_for_output(row, template_row) for row in rows] if rows else []

    if not rows:
        result["errors"].append("Manual review CSV has no data rows.")
        return result
    if len(rows) != 1:
        result["errors"].append(f"Expected exactly one data row, found {len(rows)}.")
        return result
    if not set(TEMPLATE_FIELDS).issubset(rows[0].keys()):
        result["errors"].append("Manual review CSV is missing one or more required columns.")
        return result

    row = rows[0]
    normalized: dict[str, str] = {}

    phase_value = row.get("phase", "").strip()
    add_check(
        validation_rows,
        "manual_review_phase_allowed",
        phase_value in ALLOWED_PHASES,
        "error",
        phase_value,
        sorted(ALLOWED_PHASES),
        "The manual review row must identify Phase 11O.",
    )
    if phase_value not in ALLOWED_PHASES:
        result["errors"].append("phase must be '11O' or 'Phase 11O'.")
    normalized["phase"] = phase_value

    source_type = row.get("metric_source_type", "").strip()
    add_check(
        validation_rows,
        "metric_source_type_allowed",
        source_type in ALLOWED_SOURCE_TYPES,
        "error",
        source_type,
        sorted(ALLOWED_SOURCE_TYPES),
        "Only approved manual provenance sources are allowed.",
    )
    if source_type not in ALLOWED_SOURCE_TYPES:
        result["errors"].append("metric_source_type is not allowed.")
    normalized["metric_source_type"] = source_type

    for field in ["metric_source_description", "metric_source_path_or_note", "metric_extracted_by", "metric_extracted_at_local", "reviewer_notes"]:
        normalized[field] = row.get(field, "").strip()

    for field in METRIC_FIELDS:
        value_text = row.get(field, "").strip()
        metric_value = parse_metric(value_text)
        add_check(
            validation_rows,
            f"{field}_valid",
            metric_value is not None,
            "error",
            value_text,
            "number in [0, 1]",
            f"{field} must be a numeric metric in the inclusive [0, 1] range.",
        )
        if metric_value is None:
            result["errors"].append(f"{field} must be a number in [0, 1].")
            normalized[field] = value_text
        else:
            normalized[field] = format_metric(metric_value)

    for field in BOOLEAN_FIELDS:
        value_text = row.get(field, "").strip()
        parsed = parse_bool(value_text)
        add_check(
            validation_rows,
            f"{field}_valid",
            parsed is not None,
            "error",
            value_text,
            "true or false",
            f"{field} must be explicitly set to true or false.",
        )
        if parsed is None:
            result["errors"].append(f"{field} must be true or false.")
            normalized[field] = value_text
        else:
            normalized[field] = "true" if parsed else "false"

    reviewer_decision = row.get("reviewer_decision", "").strip()
    add_check(
        validation_rows,
        "reviewer_decision_allowed",
        reviewer_decision in ALLOWED_REVIEWER_DECISIONS,
        "error",
        reviewer_decision,
        sorted(ALLOWED_REVIEWER_DECISIONS),
        "reviewer_decision must be one of the supported gate outcomes.",
    )
    if reviewer_decision not in ALLOWED_REVIEWER_DECISIONS:
        result["errors"].append("reviewer_decision is not allowed.")
    normalized["reviewer_decision"] = reviewer_decision

    result["valid"] = not result["errors"]
    result["normalized_row"] = normalized if result["valid"] else None
    result["used_rows"] = [normalized] if result["valid"] else result["used_rows"]
    return result


def build_non_execution_manifest(summary: dict[str, Any], manual_review_csv_path: Path | None) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": summary["status"],
        "manual_review_only": True,
        "manual_review_csv_path": str(manual_review_csv_path) if manual_review_csv_path else "",
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "prediction_executed": False,
        "export_executed": False,
        "checkpoint_loaded": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "weights_modified": False,
        "weights_copied_to_artifacts": False,
        "large_outputs_copied_to_artifacts": False,
        "predictions_json_used_for_metric_computation": False,
        "metrics_recomputed": False,
        "notes": (
            "Phase 11O records only manually extracted aggregate test metrics and provenance metadata. "
            "It does not rerun evaluation, load checkpoints, inspect labels/images for recomputation, "
            "or derive metrics from predictions.json."
        ),
    }


def build_template_row(phase11n_summary: dict[str, Any]) -> dict[str, str]:
    return {
        "phase": "11O",
        "metric_source_type": "",
        "metric_source_description": "Fill from visible Kaggle notebook output or copied Ultralytics console summary only.",
        "metric_source_path_or_note": phase11n_summary["eval_dir"],
        "metric_extracted_by": "",
        "metric_extracted_at_local": "",
        "test_precision": "",
        "test_recall": "",
        "test_map50": "",
        "test_map50_95": "",
        "source_confirms_test_split": "false",
        "source_confirms_model_best_pt": "false",
        "source_confirms_dataset_yaml": "false",
        "reviewer_decision": "pending_manual_review",
        "reviewer_notes": (
            "Phase 11O is review-only. Do not recompute metrics from predictions.json, images, labels, or checkpoints."
        ),
    }


def build_readme(summary: dict[str, Any], manual_review_csv_path: Path | None) -> str:
    return f"""# Phase 11O Manual Test Metric Review

Phase 11O is a strict manual-review-only gate for Phase 11M.1 test metrics.

- status = `{summary["status"]}`
- manual_metrics_available = `{summary["manual_metrics_available"]}`
- test_metrics_validated = `{summary["test_metrics_validated"]}`
- reporting_allowed = `{summary["reporting_allowed"]}`
- reviewer_decision = `{summary["reviewer_decision"]}`
- phase11n_status = `{summary["phase11n_status"]}`
- phase11n_eval_dir = `{summary["phase11n_eval_dir"]}`
- phase11n_training_dir = `{summary["phase11n_training_dir"]}`
- manual_review_csv_path = `{manual_review_csv_path if manual_review_csv_path else ""}`
- next_allowed_step = `{summary["next_allowed_step"]}`

Accepted metric provenance for this gate:

- visible Kaggle notebook output
- copied Ultralytics console summary from Phase 11M.1
- saved manual notes derived from that visible output

Still forbidden in Phase 11O:

- evaluation rerun
- inference or prediction
- checkpoint loading
- metric recomputation from `predictions.json`
- image or label inspection for metric recomputation
- copying large evaluation outputs into artifacts
"""


def normalize_row_for_output(row: dict[str, str], template_row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, template_row.get(field, "")).strip() for field in TEMPLATE_FIELDS}


def parse_metric(value_text: str) -> float | None:
    try:
        value = float(value_text)
    except ValueError:
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


def format_metric(value: float) -> str:
    return format(value, ".10g")


def parse_bool(value_text: str) -> bool | None:
    lowered = value_text.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def maybe_bool(value_text: str | None) -> bool | None:
    if value_text is None or value_text == "":
        return None
    return parse_bool(value_text)


def maybe_float(value_text: str | None) -> float | None:
    if value_text is None or value_text == "":
        return None
    try:
        return float(value_text)
    except ValueError:
        return None


def add_check(
    rows: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    severity: str,
    observed_value: Any,
    expected_value: Any,
    notes: str,
) -> None:
    rows.append(
        {
            "check_name": check_name,
            "passed": "true" if passed else "false",
            "severity": severity,
            "observed_value": json.dumps(observed_value, ensure_ascii=True)
            if isinstance(observed_value, (dict, list, tuple))
            else str(observed_value),
            "expected_value": json.dumps(expected_value, ensure_ascii=True)
            if isinstance(expected_value, (dict, list, tuple))
            else str(expected_value),
            "notes": notes,
        }
    )


def read_json_required(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return payload


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [{key: value or "" for key, value in row.items()} for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: stringify_csv_value(row.get(field, "")) for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    main()
