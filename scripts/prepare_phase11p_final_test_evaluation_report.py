"""Prepare the Phase 11P final test evaluation evidence report without execution."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11P"
DEFAULT_PHASE11N_SUMMARY = (
    "artifacts/phase11n_test_evaluation_output_collection_and_validation/"
    "phase11n_test_evaluation_output_summary.json"
)
DEFAULT_PHASE11O_SUMMARY = (
    "artifacts/phase11o_manual_test_metric_review/phase11o_manual_metric_review_summary.json"
)
DEFAULT_OUTPUT_DIR = "artifacts/phase11p_final_test_evaluation_report"
REPORT_MD_NAME = "phase11p_final_test_evaluation_report.md"
SUMMARY_JSON_NAME = "phase11p_final_test_evaluation_summary.json"
PROVENANCE_CSV_NAME = "phase11p_metric_provenance_table.csv"
MANIFEST_JSON_NAME = "phase11p_non_execution_manifest.json"
README_NAME = "README.md"
STATUS_VALIDATED = "phase11p_final_report_prepared_with_validated_manual_test_metrics"
STATUS_CAVEAT = "phase11p_final_report_prepared_with_metric_provenance_caveat"
NEXT_VALIDATED = "use_phase11p_final_report_with_manual_metric_provenance_caveat"
NEXT_CAVEAT = "recover_manual_test_metrics_via_phase11o_or_keep_phase11p_caveat_report_without_test_metric_claims"
PROVENANCE_FIELDS = [
    "metric_name",
    "value",
    "source_phase",
    "source_file",
    "source_type",
    "validation_status",
    "reporting_status",
    "caveat",
]
REQUIRED_PHASE11N_KEYS = {
    "phase",
    "status",
    "eval_dir",
    "training_dir",
    "predictions_json_available",
    "predictions_json_size_bytes",
    "confusion_matrix_available",
    "confusion_matrix_normalized_available",
    "pr_curve_available",
    "f1_curve_available",
    "p_curve_available",
    "r_curve_available",
    "validation_batch_images_count",
    "phase11l_best_epoch",
    "phase11l_best_metric_map50_95",
    "phase11l_final_epoch",
    "phase11l_final_metric_map50_95",
    "phase11l_final_metric_map50",
    "evaluation_outputs_credible",
    "provenance_caveat_carried_forward",
}
REQUIRED_PHASE11O_KEYS = {
    "phase",
    "status",
    "manual_metrics_available",
    "test_metrics_validated",
    "reporting_allowed",
    "provenance_caveat_carried_forward",
}
TRAINING_METRIC_SPECS = [
    ("phase11l_best_epoch", "best_epoch"),
    ("phase11l_best_metric_map50_95", "best_metric_map50_95"),
    ("phase11l_final_epoch", "final_epoch"),
    ("phase11l_final_metric_map50_95", "final_metric_map50_95"),
    ("phase11l_final_metric_map50", "final_metric_map50"),
]
TEST_METRIC_NAMES = ["test_precision", "test_recall", "test_map50", "test_map50_95"]
UNAVAILABLE_TEST_METRIC_CAVEAT = (
    "aggregate test metric unavailable because Phase 11M.1 did not preserve parseable "
    "results.csv/results.json/stdout metrics"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Phase 11P final test evaluation evidence report without executing anything."
    )
    parser.add_argument("--phase11n-summary", default=DEFAULT_PHASE11N_SUMMARY)
    parser.add_argument("--phase11o-summary", default=DEFAULT_PHASE11O_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_phase11p_final_test_evaluation_report(
        phase11n_summary_path=resolve_repo_path(args.phase11n_summary),
        phase11o_summary_path=resolve_repo_path(args.phase11o_summary),
        output_dir=resolve_repo_path(args.output_dir),
    )
    concise = {
        "status": summary["status"],
        "phase11o_reporting_allowed": summary["phase11o_reporting_allowed"],
        "test_metrics_reporting_status": summary["test_metrics_reporting_status"],
        "final_report_path": summary["final_report_path"],
        "next_allowed_step": summary["next_allowed_step"],
    }
    print(json.dumps(concise, indent=2))


def prepare_phase11p_final_test_evaluation_report(
    phase11n_summary_path: Path,
    phase11o_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase11n_summary = read_json_required(phase11n_summary_path, "Phase 11N summary")
    phase11o_summary = read_json_required(phase11o_summary_path, "Phase 11O summary")
    validate_required_keys(phase11n_summary, REQUIRED_PHASE11N_KEYS, phase11n_summary_path, "Phase 11N summary")
    validate_required_keys(phase11o_summary, REQUIRED_PHASE11O_KEYS, phase11o_summary_path, "Phase 11O summary")

    report_uses_validated_test_metrics = bool(
        phase11o_summary.get("reporting_allowed") is True and phase11o_summary.get("test_metrics_validated") is True
    )
    status = STATUS_VALIDATED if report_uses_validated_test_metrics else STATUS_CAVEAT
    test_metrics_reporting_status = "validated_manual_metrics_allowed" if report_uses_validated_test_metrics else "blocked_not_validated"
    next_allowed_step = NEXT_VALIDATED if report_uses_validated_test_metrics else NEXT_CAVEAT

    provenance_rows = build_metric_provenance_rows(
        phase11n_summary=phase11n_summary,
        phase11o_summary=phase11o_summary,
        phase11n_summary_path=phase11n_summary_path,
        phase11o_summary_path=phase11o_summary_path,
    )

    final_report_path = output_dir / REPORT_MD_NAME
    summary = {
        "phase": PHASE,
        "status": status,
        "phase11n_status": phase11n_summary["status"],
        "phase11o_status": phase11o_summary["status"],
        "phase11o_reporting_allowed": bool(phase11o_summary.get("reporting_allowed") is True),
        "manual_metrics_available": bool(phase11o_summary.get("manual_metrics_available") is True),
        "test_metrics_validated": bool(phase11o_summary.get("test_metrics_validated") is True),
        "test_metrics_reporting_status": test_metrics_reporting_status,
        "final_report_path": str(final_report_path),
        "phase11n_eval_dir": phase11n_summary["eval_dir"],
        "phase11n_training_dir": phase11n_summary["training_dir"],
        "phase11n_predictions_json_available": phase11n_summary["predictions_json_available"],
        "phase11n_predictions_json_size_bytes": phase11n_summary["predictions_json_size_bytes"],
        "phase11n_confusion_matrix_available": phase11n_summary["confusion_matrix_available"],
        "phase11n_confusion_matrix_normalized_available": phase11n_summary["confusion_matrix_normalized_available"],
        "phase11n_pr_curve_available": phase11n_summary["pr_curve_available"],
        "phase11n_f1_curve_available": phase11n_summary["f1_curve_available"],
        "phase11n_p_curve_available": phase11n_summary["p_curve_available"],
        "phase11n_r_curve_available": phase11n_summary["r_curve_available"],
        "phase11n_validation_batch_images_count": phase11n_summary["validation_batch_images_count"],
        "phase11n_evaluation_outputs_credible": phase11n_summary["evaluation_outputs_credible"],
        "phase11l_best_epoch": phase11n_summary["phase11l_best_epoch"],
        "phase11l_best_metric_map50_95": phase11n_summary["phase11l_best_metric_map50_95"],
        "phase11l_final_epoch": phase11n_summary["phase11l_final_epoch"],
        "phase11l_final_metric_map50_95": phase11n_summary["phase11l_final_metric_map50_95"],
        "phase11l_final_metric_map50": phase11n_summary["phase11l_final_metric_map50"],
        "validated_test_precision": phase11o_summary.get("test_precision") if report_uses_validated_test_metrics else None,
        "validated_test_recall": phase11o_summary.get("test_recall") if report_uses_validated_test_metrics else None,
        "validated_test_map50": phase11o_summary.get("test_map50") if report_uses_validated_test_metrics else None,
        "validated_test_map50_95": phase11o_summary.get("test_map50_95") if report_uses_validated_test_metrics else None,
        "metric_provenance_table_path": str(output_dir / PROVENANCE_CSV_NAME),
        "provenance_caveat_carried_forward": phase11o_summary.get(
            "provenance_caveat_carried_forward",
            phase11n_summary["provenance_caveat_carried_forward"],
        ),
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    report_text = build_report_markdown(
        summary=summary,
        provenance_rows=provenance_rows,
        phase11n_summary_path=phase11n_summary_path,
        phase11o_summary_path=phase11o_summary_path,
    )
    manifest = build_non_execution_manifest(status=status, final_report_path=final_report_path)

    final_report_path.write_text(report_text, encoding="utf-8")
    write_json(output_dir / SUMMARY_JSON_NAME, summary)
    write_csv(output_dir / PROVENANCE_CSV_NAME, provenance_rows, PROVENANCE_FIELDS)
    write_json(output_dir / MANIFEST_JSON_NAME, manifest)
    (output_dir / README_NAME).write_text(build_readme(summary), encoding="utf-8")
    return summary


def build_metric_provenance_rows(
    phase11n_summary: dict[str, Any],
    phase11o_summary: dict[str, Any],
    phase11n_summary_path: Path,
    phase11o_summary_path: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for summary_key, metric_name in TRAINING_METRIC_SPECS:
        value = phase11n_summary.get(summary_key)
        rows.append(
            {
                "metric_name": metric_name,
                "value": stringify_value(value),
                "source_phase": "11L via 11N",
                "source_file": str(phase11n_summary_path),
                "source_type": "validated_local_training_output_integrity_summary",
                "validation_status": "validated_by_phase11l_local_output_integrity",
                "reporting_status": "allowed_with_training_output_provenance_caveat",
                "caveat": phase11n_summary["provenance_caveat_carried_forward"],
            }
        )

    report_uses_validated_test_metrics = bool(
        phase11o_summary.get("reporting_allowed") is True and phase11o_summary.get("test_metrics_validated") is True
    )
    for metric_name in TEST_METRIC_NAMES:
        if report_uses_validated_test_metrics:
            rows.append(
                {
                    "metric_name": metric_name,
                    "value": stringify_value(phase11o_summary.get(metric_name)),
                    "source_phase": "11O",
                    "source_file": str(phase11o_summary_path),
                    "source_type": "validated_manual_metric_review",
                    "validation_status": "validated_manual_metric_provenance",
                    "reporting_status": "allowed_with_manual_metric_provenance_caveat",
                    "caveat": "manually extracted metric, not machine-parsed from results.csv/results.json/stdout",
                }
            )
        else:
            rows.append(
                {
                    "metric_name": metric_name,
                    "value": "",
                    "source_phase": "11O",
                    "source_file": str(phase11o_summary_path),
                    "source_type": "manual_metric_review_gate_pending_or_blocked",
                    "validation_status": "not_validated",
                    "reporting_status": "blocked",
                    "caveat": UNAVAILABLE_TEST_METRIC_CAVEAT,
                }
            )
    return rows


def build_report_markdown(
    summary: dict[str, Any],
    provenance_rows: list[dict[str, str]],
    phase11n_summary_path: Path,
    phase11o_summary_path: Path,
) -> str:
    test_metrics_allowed = summary["phase11o_reporting_allowed"]
    lines: list[str] = []
    lines.append("# Phase 11P Final Test Evaluation Evidence Report")
    lines.append("")
    lines.append("## Section 1: Scope and status")
    lines.append("")
    lines.append(f"- Phase 11P status: `{summary['status']}`")
    lines.append("- Scope: report-only consolidation for Phase 11J through Phase 11O.")
    lines.append("- Inputs:")
    lines.append(f"  - Phase 11N summary: `{phase11n_summary_path}`")
    lines.append(f"  - Phase 11O summary: `{phase11o_summary_path}`")
    lines.append("- No training, evaluation, inference, prediction, export, checkpoint loading, or metric recomputation was performed by Phase 11P.")
    lines.append("")
    lines.append("## Section 2: Training output integrity summary")
    lines.append("")
    lines.append("- The training output integrity record carried forward from Phase 11L remains available and reportable with provenance caveat.")
    lines.append(f"- Best epoch from Phase 11L: `{summary['phase11l_best_epoch']}`")
    lines.append(f"- Best `mAP50-95` from Phase 11L: `{summary['phase11l_best_metric_map50_95']}`")
    lines.append(f"- Final epoch from Phase 11L: `{summary['phase11l_final_epoch']}`")
    lines.append(f"- Final `mAP50-95` from Phase 11L: `{summary['phase11l_final_metric_map50_95']}`")
    lines.append(f"- Final `mAP50` from Phase 11L: `{summary['phase11l_final_metric_map50']}`")
    lines.append("- These are training-output metrics validated by local output integrity checks, not new test metrics from Phase 11M.1.")
    lines.append("")
    lines.append("## Section 3: Test evaluation output inventory")
    lines.append("")
    lines.append(f"- Evaluation directory from Phase 11N: `{summary['phase11n_eval_dir']}`")
    lines.append(f"- Training directory linked by Phase 11N: `{summary['phase11n_training_dir']}`")
    lines.append(f"- `predictions.json` available: `{summary['phase11n_predictions_json_available']}`")
    lines.append(f"- `predictions.json` size bytes: `{summary['phase11n_predictions_json_size_bytes']}`")
    lines.append(f"- `confusion_matrix.png` available: `{summary['phase11n_confusion_matrix_available']}`")
    lines.append(
        f"- `confusion_matrix_normalized.png` available: `{summary['phase11n_confusion_matrix_normalized_available']}`"
    )
    lines.append(f"- `PR` curve available: `{summary['phase11n_pr_curve_available']}`")
    lines.append(f"- `F1` curve available: `{summary['phase11n_f1_curve_available']}`")
    lines.append(f"- `P` curve available: `{summary['phase11n_p_curve_available']}`")
    lines.append(f"- `R` curve available: `{summary['phase11n_r_curve_available']}`")
    lines.append(f"- Validation batch image count: `{summary['phase11n_validation_batch_images_count']}`")
    lines.append(f"- Evaluation outputs credible as files: `{summary['phase11n_evaluation_outputs_credible']}`")
    lines.append("")
    lines.append("## Section 4: Test metric provenance status")
    lines.append("")
    if test_metrics_allowed:
        lines.append("- Phase 11O allows reporting of manually validated aggregate test metrics with explicit provenance caveat.")
        lines.append(
            f"- Validated manual test precision: `{summary['validated_test_precision']}`"
        )
        lines.append(
            f"- Validated manual test recall: `{summary['validated_test_recall']}`"
        )
        lines.append(
            f"- Validated manual test `mAP50`: `{summary['validated_test_map50']}`"
        )
        lines.append(
            f"- Validated manual test `mAP50-95`: `{summary['validated_test_map50_95']}`"
        )
        lines.append("- These values remain manually extracted, not machine-parsed from preserved `results.csv`, `results.json`, or stdout artifacts.")
    else:
        lines.append(
            "- Aggregate test metrics were not available from a parseable source and were not manually validated in Phase 11O."
        )
        lines.append("- Evaluation outputs are credible as files, but aggregate metric reporting remains blocked.")
        lines.append("- No numeric claim about test precision, recall, `mAP50`, or `mAP50-95` is made in this report.")
    lines.append("")
    lines.append("## Section 5: What can be reported")
    lines.append("")
    lines.append("- The Phase 11L training output integrity summary can be reported with provenance caveat.")
    lines.append("- The Phase 11N evaluation output directory can be reported as credible by file inventory and artifact presence.")
    if test_metrics_allowed:
        lines.append("- The Phase 11O manually validated aggregate test metrics can be reported with manual-provenance caveat.")
    else:
        lines.append("- The current final report can state that test evaluation outputs exist, but aggregate test metrics remain unavailable or not validated for reporting.")
    lines.append("")
    lines.append("## Section 6: What cannot be claimed")
    lines.append("")
    if test_metrics_allowed:
        lines.append("- The validated test metrics should not be described as machine-parsed or directly reproduced from preserved evaluation result files.")
    else:
        lines.append("- No wording such as `test mAP achieved ...` is allowed in the current state.")
        lines.append("- No numeric test precision, recall, `mAP50`, or `mAP50-95` claim is allowed.")
    lines.append("- This report does not prove Kaggle execution provenance beyond the carried-forward caveat.")
    lines.append("- This report does not recompute any metric from `predictions.json` or from images, labels, or checkpoints.")
    lines.append("")
    lines.append("## Section 7: Caveats and next recommended action")
    lines.append("")
    lines.append(f"- Carried-forward provenance caveat: {summary['provenance_caveat_carried_forward']}")
    if test_metrics_allowed:
        lines.append("- The remaining metric caveat is that test metrics were manually extracted and validated in Phase 11O rather than preserved as machine-readable result files.")
        lines.append("- Recommended next action: use this report with the manual-metric provenance caveat intact.")
    else:
        lines.append("- The remaining metric caveat is that Phase 11M.1 did not preserve parseable aggregate test metric artifacts.")
        lines.append("- Recommended next action: recover visible Kaggle notebook metrics into Phase 11O or keep using this report without numeric test metric claims.")
    lines.append("")
    lines.append("## Section 8: Non-execution and non-mutation guarantees")
    lines.append("")
    lines.append("- `training_executed = false`")
    lines.append("- `evaluation_executed = false`")
    lines.append("- `inference_executed = false`")
    lines.append("- `prediction_executed = false`")
    lines.append("- `export_executed = false`")
    lines.append("- `checkpoint_loaded = false`")
    lines.append("- `dataset_mutated = false`")
    lines.append("- `labels_mutated = false`")
    lines.append("- `weights_modified = false`")
    lines.append("- `weights_copied_to_artifacts = false`")
    lines.append("- `large_outputs_copied_to_artifacts = false`")
    lines.append("- `predictions_json_used_for_metric_computation = false`")
    lines.append("- `metrics_recomputed = false`")
    lines.append("")
    lines.append("## Metric provenance table")
    lines.append("")
    lines.append("| metric_name | value | source_phase | validation_status | reporting_status |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in provenance_rows:
        lines.append(
            f"| {row['metric_name']} | {row['value'] or ' '} | {row['source_phase']} | {row['validation_status']} | {row['reporting_status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_non_execution_manifest(status: str, final_report_path: Path) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": status,
        "final_report_path": str(final_report_path),
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
            "Phase 11P is report-only. It consolidates existing metadata from Phase 11N and Phase 11O "
            "without recomputing metrics or copying large runtime outputs."
        ),
    }


def build_readme(summary: dict[str, Any]) -> str:
    return f"""# Phase 11P Final Test Evaluation Report

This artifact bundle contains the report-only Phase 11P consolidation for Phase 11J through Phase 11O.

- status = `{summary["status"]}`
- phase11o_reporting_allowed = `{summary["phase11o_reporting_allowed"]}`
- test_metrics_reporting_status = `{summary["test_metrics_reporting_status"]}`
- final_report_path = `{summary["final_report_path"]}`
- next_allowed_step = `{summary["next_allowed_step"]}`

Contents:

- `phase11p_final_test_evaluation_report.md`
- `phase11p_final_test_evaluation_summary.json`
- `phase11p_metric_provenance_table.csv`
- `phase11p_non_execution_manifest.json`

This folder must remain small metadata only. No large evaluation outputs, images, or weights should be copied here.
"""


def validate_required_keys(payload: dict[str, Any], required_keys: set[str], path: Path, label: str) -> None:
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise SystemExit(f"{label} is missing required keys at {path}: {', '.join(missing)}")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: stringify_value(row.get(field, "")) for field in fieldnames})


def stringify_value(value: Any) -> str:
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
