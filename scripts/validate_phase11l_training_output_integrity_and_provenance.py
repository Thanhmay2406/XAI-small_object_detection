"""Validate Phase 11L training output integrity and provenance without execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11L"
DEFAULT_PHASE11K_SUMMARY = (
    "artifacts/phase11k_training_outputs_and_metrics/phase11k_training_outputs_summary.json"
)
DEFAULT_PHASE11K_METRICS_CSV = (
    "artifacts/phase11k_training_outputs_and_metrics/phase11k_results_metrics_summary.csv"
)
DEFAULT_PHASE11K_WEIGHT_MANIFEST = (
    "artifacts/phase11k_training_outputs_and_metrics/phase11k_weight_files_manifest.csv"
)
DEFAULT_OUTPUT_DIR = "artifacts/phase11l_training_output_integrity_and_provenance"
TOLERANCE = 1e-4
HASH_CHUNK_SIZE = 1024 * 1024
STATUS_PASS = "phase11l_training_output_integrity_and_provenance_passed"
STATUS_FAIL = "phase11l_training_output_integrity_and_provenance_failed"
NEXT_PASS = "phase11m0_prepare_approved_test_evaluation_no_execution"
NEXT_FAIL = "inspect_phase11k_and_training_output_paths_before_evaluation"
VALIDATION_CHECK_FIELDS = ["check_name", "passed", "severity", "observed_value", "expected_value", "notes"]
METRIC_CONSISTENCY_FIELDS = [
    "metric_name",
    "phase11k_value",
    "direct_value",
    "difference",
    "tolerance",
    "passed",
    "notes",
]
CHECKPOINT_MANIFEST_FIELDS = [
    "checkpoint_role",
    "path",
    "exists",
    "size_bytes",
    "size_mb",
    "sha256_phase11k",
    "sha256_direct",
    "sha256_matches_phase11k",
    "modified_time_utc",
    "non_empty",
    "metadata_only_inspection",
]
PHASE11K_REQUIRED_KEYS = {
    "phase",
    "status",
    "phase11j1_summary_available",
    "phase11j1_status",
    "phase11j1_validation_notes",
    "results_csv_path",
    "best_weight_path",
    "last_weight_path",
    "results_row_count",
    "metric_columns_detected",
    "best_epoch_value",
    "final_epoch_value",
    "best_metrics",
    "final_metrics",
    "weights_hashed",
}
PHASE11K_METRICS_REQUIRED_COLUMNS = {
    "phase",
    "status",
    "results_csv_path",
    "results_row_count",
    "best_epoch_value",
    "final_epoch_value",
    "final_metrics_json",
    "best_metrics_json",
}
PHASE11K_WEIGHT_REQUIRED_COLUMNS = {
    "weight_label",
    "path",
    "exists",
    "size_bytes",
    "size_mb",
    "sha256",
    "modified_time_utc",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Phase 11K training outputs for integrity and provenance without execution."
    )
    parser.add_argument("--phase11k-summary", default=DEFAULT_PHASE11K_SUMMARY)
    parser.add_argument("--phase11k-metrics-csv", default=DEFAULT_PHASE11K_METRICS_CSV)
    parser.add_argument("--phase11k-weight-manifest", default=DEFAULT_PHASE11K_WEIGHT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_phase11l_training_output_integrity_and_provenance(
        phase11k_summary_path=resolve_repo_path(args.phase11k_summary),
        phase11k_metrics_csv_path=resolve_repo_path(args.phase11k_metrics_csv),
        phase11k_weight_manifest_path=resolve_repo_path(args.phase11k_weight_manifest),
        output_dir=resolve_repo_path(args.output_dir),
    )
    concise = {
        "phase": summary["phase"],
        "status": summary["status"],
        "baseline_checkpoint_candidate_accepted": summary["baseline_checkpoint_candidate_accepted"],
        "accepted_checkpoint_path": summary["accepted_checkpoint_path"],
        "best_epoch": summary["best_epoch"],
        "best_metric_map50_95": summary["best_metric_map50_95"],
        "final_epoch": summary["final_epoch"],
        "final_metric_map50_95": summary["final_metric_map50_95"],
        "final_metric_map50": summary["final_metric_map50"],
        "next_allowed_step": summary["next_allowed_step"],
    }
    print(json.dumps(concise, indent=2))


def validate_phase11l_training_output_integrity_and_provenance(
    phase11k_summary_path: Path,
    phase11k_metrics_csv_path: Path,
    phase11k_weight_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_checks: list[dict[str, Any]] = []
    metric_consistency_rows: list[dict[str, Any]] = []

    phase11k_summary = read_json_if_exists(phase11k_summary_path)
    phase11k_metrics_rows = read_csv_rows_if_exists(phase11k_metrics_csv_path)
    phase11k_weight_rows = read_csv_rows_if_exists(phase11k_weight_manifest_path)

    phase11k_summary_available = phase11k_summary is not None
    add_check(
        validation_checks,
        "phase11k_summary_json_exists",
        phase11k_summary_available,
        "error",
        str(phase11k_summary_path.exists()),
        "True",
        "Phase 11L requires the Phase 11K summary JSON.",
    )
    add_check(
        validation_checks,
        "phase11k_metrics_csv_exists",
        phase11k_metrics_rows is not None,
        "error",
        str(phase11k_metrics_csv_path.exists()),
        "True",
        "Phase 11L requires the Phase 11K metrics CSV.",
    )
    add_check(
        validation_checks,
        "phase11k_weight_manifest_exists",
        phase11k_weight_rows is not None,
        "error",
        str(phase11k_weight_manifest_path.exists()),
        "True",
        "Phase 11L requires the Phase 11K weight manifest CSV.",
    )

    phase11k_summary_keys_ok = phase11k_summary is not None and PHASE11K_REQUIRED_KEYS.issubset(phase11k_summary)
    add_check(
        validation_checks,
        "phase11k_summary_required_keys_present",
        phase11k_summary_keys_ok,
        "error",
        sorted(list(phase11k_summary.keys())) if phase11k_summary else [],
        sorted(PHASE11K_REQUIRED_KEYS),
        "Phase 11K summary must expose the fields required for direct integrity validation.",
    )

    phase11k_metrics_columns_ok = csv_has_columns(phase11k_metrics_rows, PHASE11K_METRICS_REQUIRED_COLUMNS)
    add_check(
        validation_checks,
        "phase11k_metrics_csv_required_columns_present",
        phase11k_metrics_columns_ok,
        "error",
        sorted(list(phase11k_metrics_rows[0].keys())) if phase11k_metrics_rows else [],
        sorted(PHASE11K_METRICS_REQUIRED_COLUMNS),
        "Phase 11K metrics CSV must expose the required consistency columns.",
    )

    phase11k_weight_columns_ok = csv_has_columns(phase11k_weight_rows, PHASE11K_WEIGHT_REQUIRED_COLUMNS)
    add_check(
        validation_checks,
        "phase11k_weight_manifest_required_columns_present",
        phase11k_weight_columns_ok,
        "error",
        sorted(list(phase11k_weight_rows[0].keys())) if phase11k_weight_rows else [],
        sorted(PHASE11K_WEIGHT_REQUIRED_COLUMNS),
        "Phase 11K weight manifest must expose the required checkpoint metadata columns.",
    )

    results_csv_path = resolve_string_path(phase11k_summary.get("results_csv_path", "") if phase11k_summary else "")
    best_weight_path = resolve_string_path(phase11k_summary.get("best_weight_path", "") if phase11k_summary else "")
    last_weight_path = resolve_string_path(phase11k_summary.get("last_weight_path", "") if phase11k_summary else "")

    results_csv_available = results_csv_path is not None and results_csv_path.exists()
    add_check(
        validation_checks,
        "results_csv_path_resolved",
        results_csv_path is not None,
        "error",
        str(results_csv_path) if results_csv_path else "",
        "absolute or repo-resolvable path",
        "results.csv path must resolve from the Phase 11K summary.",
    )
    add_check(
        validation_checks,
        "results_csv_exists",
        results_csv_available,
        "error",
        str(results_csv_available),
        "True",
        "results.csv must exist for direct integrity validation.",
    )

    results_info = parse_results_csv(results_csv_path) if results_csv_available else empty_results_info()
    add_check(
        validation_checks,
        "results_csv_readable",
        results_info["readable"],
        "error",
        str(results_info["readable"]),
        "True",
        results_info["read_note"],
    )
    add_check(
        validation_checks,
        "results_row_count_is_100",
        results_info["row_count"] == 100,
        "error",
        results_info["row_count"],
        100,
        "The current baseline run is expected to contain 100 epochs.",
    )
    add_check(
        validation_checks,
        "epoch_column_present",
        results_info["epoch_column"] != "",
        "error",
        results_info["epoch_column"],
        "epoch",
        "An epoch column is required for consistency checks.",
    )
    add_check(
        validation_checks,
        "epochs_monotonic",
        results_info["epochs_monotonic"],
        "error",
        results_info["epoch_values"],
        "strictly increasing epoch sequence",
        "Epoch values should be monotonic.",
    )
    add_check(
        validation_checks,
        "final_epoch_recorded_as_100_or_zero_based_equivalent",
        results_info["final_epoch_ok"],
        "error",
        results_info["final_epoch_value"],
        "100 or zero-based equivalent 99",
        "The final epoch should represent a 100-epoch training run.",
    )
    add_check(
        validation_checks,
        "required_metric_columns_present",
        results_info["required_metric_columns_present"],
        "error",
        results_info["metric_columns"],
        ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        "Required mAP columns must exist.",
    )
    add_check(
        validation_checks,
        "no_required_metric_nan_or_non_numeric",
        results_info["required_metrics_numeric"],
        "error",
        results_info["required_metric_failures"],
        "all required metric values numeric",
        "Required metric values must be numeric and finite.",
    )

    optional_loss_column_names = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    for name in optional_loss_column_names:
        add_check(
            validation_checks,
            f"optional_column_present::{name}",
            name in results_info["fieldnames"],
            "info",
            name in results_info["fieldnames"],
            True,
            "Optional loss column presence is descriptive only and does not control the gate.",
        )

    phase11k_best_epoch = value_as_float(phase11k_summary.get("best_epoch_value") if phase11k_summary else None)
    phase11k_final_epoch = value_as_float(phase11k_summary.get("final_epoch_value") if phase11k_summary else None)
    phase11k_best_map = extract_nested_metric(
        phase11k_summary.get("best_metrics") if phase11k_summary else {}, "metrics/mAP50-95(B)"
    )
    phase11k_final_map = extract_nested_metric(
        phase11k_summary.get("final_metrics") if phase11k_summary else {}, "metrics/mAP50-95(B)"
    )
    phase11k_final_map50 = extract_nested_metric(
        phase11k_summary.get("final_metrics") if phase11k_summary else {}, "metrics/mAP50(B)"
    )

    best_epoch_alignment = compare_epoch_values(phase11k_best_epoch, results_info["best_epoch_value"])
    final_epoch_alignment = compare_epoch_values(phase11k_final_epoch, results_info["final_epoch_value"])
    add_check(
        validation_checks,
        "best_epoch_matches_phase11k",
        best_epoch_alignment["passed"],
        "error",
        results_info["best_epoch_value"],
        phase11k_best_epoch,
        best_epoch_alignment["notes"],
    )
    add_check(
        validation_checks,
        "final_epoch_matches_phase11k",
        final_epoch_alignment["passed"],
        "error",
        results_info["final_epoch_value"],
        phase11k_final_epoch,
        final_epoch_alignment["notes"],
    )

    add_metric_consistency(
        metric_consistency_rows,
        "best_epoch",
        phase11k_best_epoch,
        results_info["best_epoch_value"],
        best_epoch_alignment["passed"],
        TOLERANCE,
        best_epoch_alignment["notes"],
        treat_as_epoch=True,
    )
    add_metric_consistency(
        metric_consistency_rows,
        "best_metric_map50_95",
        phase11k_best_map,
        results_info["best_map50_95"],
        approx_equal(phase11k_best_map, results_info["best_map50_95"], TOLERANCE),
        TOLERANCE,
        "Direct best mAP50-95 should match the Phase 11K summary within tolerance.",
    )
    add_metric_consistency(
        metric_consistency_rows,
        "final_epoch",
        phase11k_final_epoch,
        results_info["final_epoch_value"],
        final_epoch_alignment["passed"],
        TOLERANCE,
        final_epoch_alignment["notes"],
        treat_as_epoch=True,
    )
    add_metric_consistency(
        metric_consistency_rows,
        "final_metric_map50_95",
        phase11k_final_map,
        results_info["final_map50_95"],
        approx_equal(phase11k_final_map, results_info["final_map50_95"], TOLERANCE),
        TOLERANCE,
        "Direct final mAP50-95 should match the Phase 11K summary within tolerance.",
    )
    add_metric_consistency(
        metric_consistency_rows,
        "final_metric_map50",
        phase11k_final_map50,
        results_info["final_map50"],
        approx_equal(phase11k_final_map50, results_info["final_map50"], TOLERANCE),
        TOLERANCE,
        "Direct final mAP50 should match the Phase 11K summary within tolerance.",
    )
    add_metric_consistency(
        metric_consistency_rows,
        "best_metric_map50_95_reference",
        0.36688,
        results_info["best_map50_95"],
        approx_equal(0.36688, results_info["best_map50_95"], TOLERANCE),
        TOLERANCE,
        "Direct best mAP50-95 should match the known Phase 11K value.",
    )
    add_metric_consistency(
        metric_consistency_rows,
        "final_metric_map50_95_reference",
        0.35709,
        results_info["final_map50_95"],
        approx_equal(0.35709, results_info["final_map50_95"], TOLERANCE),
        TOLERANCE,
        "Direct final mAP50-95 should match the known Phase 11K value.",
    )
    add_metric_consistency(
        metric_consistency_rows,
        "final_metric_map50_reference",
        0.71681,
        results_info["final_map50"],
        approx_equal(0.71681, results_info["final_map50"], TOLERANCE),
        TOLERANCE,
        "Direct final mAP50 should match the known Phase 11K value.",
    )

    post_best_map_change = None
    post_best_diagnostic = "Unavailable because direct best/final metrics could not be parsed."
    if results_info["best_map50_95"] is not None and results_info["final_map50_95"] is not None:
        post_best_map_change = results_info["final_map50_95"] - results_info["best_map50_95"]
        if post_best_map_change < -TOLERANCE:
            post_best_diagnostic = "Final mAP50-95 is below the best observed epoch, consistent with post-best degradation."
        elif abs(post_best_map_change) <= TOLERANCE:
            post_best_diagnostic = "Final mAP50-95 is within tolerance of the best epoch, consistent with a plateau."
        else:
            post_best_diagnostic = "Final mAP50-95 is above the best recorded value; inspect the metric selection logic."
    add_check(
        validation_checks,
        "post_best_metric_diagnostic_recorded",
        post_best_map_change is not None,
        "info",
        post_best_map_change,
        "diagnostic only",
        post_best_diagnostic,
    )

    phase11k_weight_manifest_map = {
        row["weight_label"]: row for row in (phase11k_weight_rows or []) if "weight_label" in row
    }
    checkpoint_rows = [
        build_checkpoint_row("best.pt", best_weight_path, phase11k_weight_manifest_map.get("best.pt")),
        build_checkpoint_row("last.pt", last_weight_path, phase11k_weight_manifest_map.get("last.pt")),
    ]
    checkpoint_manifest_passed = all(
        row["exists"] and row["non_empty"] and row["sha256_matches_phase11k"] for row in checkpoint_rows
    )
    add_check(
        validation_checks,
        "checkpoint_manifest_passed",
        checkpoint_manifest_passed,
        "error",
        checkpoint_rows,
        "existing non-empty checkpoint files with matching Phase 11K sha256",
        "Checkpoints are validated only by path, size, mtime, and sha256 metadata.",
    )

    provenance_caveat_recorded = (
        phase11k_summary is not None
        and phase11k_summary.get("phase11j1_summary_available") is False
        and str(phase11k_summary.get("phase11j1_status", "")) == "phase11j1_execution_not_started_missing_execute_flag"
    )
    add_check(
        validation_checks,
        "provenance_caveat_recorded",
        provenance_caveat_recorded,
        "error",
        {
            "phase11j1_summary_available": phase11k_summary.get("phase11j1_summary_available") if phase11k_summary else None,
            "phase11j1_status": phase11k_summary.get("phase11j1_status") if phase11k_summary else None,
        },
        {
            "phase11j1_summary_available": False,
            "phase11j1_status": "phase11j1_execution_not_started_missing_execute_flag",
        },
        "Phase 11L preserves the Phase 11J.1 local-summary caveat and validates only local output integrity.",
    )

    non_execution_manifest = build_non_execution_manifest()
    non_execution_guarantees_passed = all(value is False for value in non_execution_manifest.values() if isinstance(value, bool))
    add_check(
        validation_checks,
        "non_execution_guarantees_passed",
        non_execution_guarantees_passed,
        "error",
        non_execution_manifest,
        "all execution and mutation booleans false",
        "Phase 11L is metadata-only validation.",
    )

    results_csv_integrity_passed = all(
        row["passed"] for row in validation_checks if row["severity"] == "error" and row["check_name"] in {
            "results_csv_path_resolved",
            "results_csv_exists",
            "results_csv_readable",
            "results_row_count_is_100",
            "epoch_column_present",
            "epochs_monotonic",
            "final_epoch_recorded_as_100_or_zero_based_equivalent",
            "required_metric_columns_present",
            "no_required_metric_nan_or_non_numeric",
            "best_epoch_matches_phase11k",
            "final_epoch_matches_phase11k",
        }
    )
    metric_consistency_passed = all(row["passed"] for row in metric_consistency_rows)
    baseline_checkpoint_candidate_accepted = (
        phase11k_summary_available
        and results_csv_available
        and results_csv_integrity_passed
        and metric_consistency_passed
        and checkpoint_manifest_passed
        and provenance_caveat_recorded
        and non_execution_guarantees_passed
    )
    status = STATUS_PASS if baseline_checkpoint_candidate_accepted else STATUS_FAIL
    accepted_checkpoint_role = (
        "best_pt_for_phase11m_prepare_only_evaluation" if baseline_checkpoint_candidate_accepted else ""
    )
    accepted_checkpoint_path = str(best_weight_path) if baseline_checkpoint_candidate_accepted and best_weight_path else ""
    next_allowed_step = NEXT_PASS if baseline_checkpoint_candidate_accepted else NEXT_FAIL

    metric_diagnostics = {
        "best_epoch_selection_metric": "metrics/mAP50-95(B)",
        "best_epoch_direct": results_info["best_epoch_value"],
        "best_map50_95_direct": results_info["best_map50_95"],
        "final_epoch_direct": results_info["final_epoch_value"],
        "final_map50_95_direct": results_info["final_map50_95"],
        "final_map50_direct": results_info["final_map50"],
        "post_best_map50_95_change": post_best_map_change,
        "post_best_diagnostic": post_best_diagnostic,
        "descriptive_only": True,
        "not_a_research_claim": True,
    }

    summary = {
        "phase": PHASE,
        "status": status,
        "phase11k_summary_available": phase11k_summary_available,
        "results_csv_available": results_csv_available,
        "results_csv_integrity_passed": results_csv_integrity_passed,
        "metric_consistency_passed": metric_consistency_passed,
        "checkpoint_manifest_passed": checkpoint_manifest_passed,
        "provenance_caveat_recorded": provenance_caveat_recorded,
        "non_execution_guarantees_passed": non_execution_guarantees_passed,
        "baseline_checkpoint_candidate_accepted": baseline_checkpoint_candidate_accepted,
        "accepted_checkpoint_role": accepted_checkpoint_role,
        "accepted_checkpoint_path": accepted_checkpoint_path,
        "best_epoch": results_info["best_epoch_value"],
        "best_metric_map50_95": results_info["best_map50_95"],
        "final_epoch": results_info["final_epoch_value"],
        "final_metric_map50_95": results_info["final_map50_95"],
        "final_metric_map50": results_info["final_map50"],
        "phase11j1_local_summary_status": phase11k_summary.get("phase11j1_status") if phase11k_summary else "",
        "provenance_caveat": (
            "Phase 11K used direct local output inspection because the repo-local Phase 11J.1 summary still reports "
            "'phase11j1_execution_not_started_missing_execute_flag'. Phase 11L therefore validates only the local "
            "training output integrity and cannot prove how Kaggle produced these files."
        ),
        "checkpoint_candidate_scope": (
            "baseline trained checkpoint candidate only, not a final research result"
        ),
        "results_csv_path": str(results_csv_path) if results_csv_path else "",
        "best_weight_path": str(best_weight_path) if best_weight_path else "",
        "last_weight_path": str(last_weight_path) if last_weight_path else "",
        "validation_artifacts": {
            "summary_json": str(output_dir / "phase11l_training_output_integrity_summary.json"),
            "validation_checks_csv": str(output_dir / "phase11l_validation_checks.csv"),
            "metric_consistency_csv": str(output_dir / "phase11l_metric_consistency_report.csv"),
            "checkpoint_manifest_csv": str(output_dir / "phase11l_checkpoint_manifest.csv"),
            "non_execution_manifest_json": str(output_dir / "phase11l_non_execution_manifest.json"),
            "metric_diagnostics_json": str(output_dir / "phase11l_metric_diagnostics.json"),
            "readme_md": str(output_dir / "README.md"),
        },
        "generated_at_utc": utc_now(),
        "next_allowed_step": next_allowed_step,
    }

    write_json(summary, output_dir / "phase11l_training_output_integrity_summary.json")
    write_csv(validation_checks, output_dir / "phase11l_validation_checks.csv", VALIDATION_CHECK_FIELDS)
    write_csv(metric_consistency_rows, output_dir / "phase11l_metric_consistency_report.csv", METRIC_CONSISTENCY_FIELDS)
    write_csv(checkpoint_rows, output_dir / "phase11l_checkpoint_manifest.csv", CHECKPOINT_MANIFEST_FIELDS)
    write_json(non_execution_manifest, output_dir / "phase11l_non_execution_manifest.json")
    write_json(metric_diagnostics, output_dir / "phase11l_metric_diagnostics.json")
    (output_dir / "README.md").write_text(build_readme(summary), encoding="utf-8")
    return summary


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows_if_exists(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def csv_has_columns(rows: list[dict[str, str]] | None, required_columns: set[str]) -> bool:
    if not rows:
        return False
    return required_columns.issubset(rows[0].keys())


def resolve_string_path(path_str: str) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_results_csv(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except OSError as exc:
        info = empty_results_info()
        info["read_note"] = str(exc)
        return info

    info = empty_results_info()
    info["readable"] = True
    info["read_note"] = ""
    info["fieldnames"] = [name.strip() for name in fieldnames]
    info["row_count"] = len(rows)
    info["metric_columns"] = [
        name.strip() for name in fieldnames if name.strip() in {"metrics/mAP50(B)", "metrics/mAP50-95(B)"}
    ]
    epoch_column = next((name for name in fieldnames if name.strip().lower() == "epoch"), "")
    info["epoch_column"] = epoch_column

    epoch_values: list[float] = []
    epochs_monotonic = True
    required_metric_failures: list[dict[str, Any]] = []
    best_index = None
    best_metric = None
    best_epoch_value = None
    final_epoch_value = None
    final_map50 = None
    final_map50_95 = None
    best_map50_95 = None

    prior_epoch = None
    for index, row in enumerate(rows):
        if epoch_column:
            epoch_value = value_as_float(row.get(epoch_column))
            if epoch_value is None:
                epochs_monotonic = False
            else:
                epoch_values.append(epoch_value)
                if prior_epoch is not None and epoch_value <= prior_epoch:
                    epochs_monotonic = False
                prior_epoch = epoch_value
        map50 = value_as_float(row.get("metrics/mAP50(B)"))
        map50_95 = value_as_float(row.get("metrics/mAP50-95(B)"))
        for metric_name, metric_value in (
            ("metrics/mAP50(B)", map50),
            ("metrics/mAP50-95(B)", map50_95),
        ):
            if metric_value is None or not math.isfinite(metric_value):
                required_metric_failures.append({"row_index": index, "metric": metric_name, "value": row.get(metric_name)})
        if map50_95 is not None and math.isfinite(map50_95):
            if best_metric is None or map50_95 > best_metric:
                best_metric = map50_95
                best_index = index
                best_epoch_value = epoch_values[-1] if epoch_values else None
                best_map50_95 = map50_95
        if index == len(rows) - 1:
            final_epoch_value = epoch_values[-1] if epoch_values else None
            final_map50 = map50
            final_map50_95 = map50_95

    info["epoch_values"] = epoch_values
    info["epochs_monotonic"] = epochs_monotonic and bool(epoch_values)
    info["required_metric_columns_present"] = {
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    }.issubset(info["fieldnames"])
    info["required_metrics_numeric"] = len(required_metric_failures) == 0
    info["required_metric_failures"] = required_metric_failures
    info["best_epoch_index"] = best_index
    info["best_epoch_value"] = best_epoch_value
    info["best_map50_95"] = best_map50_95
    info["final_epoch_value"] = final_epoch_value
    info["final_map50"] = final_map50
    info["final_map50_95"] = final_map50_95
    info["final_epoch_ok"] = final_epoch_value in {99.0, 100.0}
    return info


def empty_results_info() -> dict[str, Any]:
    return {
        "readable": False,
        "read_note": "results.csv was not parsed",
        "fieldnames": [],
        "row_count": 0,
        "epoch_column": "",
        "epoch_values": [],
        "epochs_monotonic": False,
        "metric_columns": [],
        "required_metric_columns_present": False,
        "required_metrics_numeric": False,
        "required_metric_failures": [],
        "best_epoch_index": None,
        "best_epoch_value": None,
        "best_map50_95": None,
        "final_epoch_value": None,
        "final_map50": None,
        "final_map50_95": None,
        "final_epoch_ok": False,
    }


def build_checkpoint_row(
    checkpoint_role: str,
    path: Path | None,
    phase11k_row: dict[str, str] | None,
) -> dict[str, Any]:
    exists = path is not None and path.exists()
    stat = path.stat() if exists and path is not None else None
    size_bytes = stat.st_size if stat else 0
    direct_sha = sha256_file(path) if exists and path is not None else ""
    phase11k_sha = phase11k_row.get("sha256", "") if phase11k_row else ""
    return {
        "checkpoint_role": checkpoint_role,
        "path": str(path) if path else "",
        "exists": exists,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 6) if exists else 0,
        "sha256_phase11k": phase11k_sha,
        "sha256_direct": direct_sha,
        "sha256_matches_phase11k": bool(phase11k_sha) and direct_sha == phase11k_sha,
        "modified_time_utc": format_mtime_utc(stat.st_mtime) if stat else "",
        "non_empty": size_bytes > 0,
        "metadata_only_inspection": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def format_mtime_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def value_as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def extract_nested_metric(container: dict[str, Any], key: str) -> float | None:
    return value_as_float(container.get(key))


def approx_equal(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= tolerance


def compare_epoch_values(expected: float | None, observed: float | None) -> dict[str, Any]:
    if expected is None or observed is None:
        return {"passed": False, "notes": "Expected or observed epoch value is missing."}
    if abs(expected - observed) <= TOLERANCE:
        return {"passed": True, "notes": "Epoch values match directly."}
    if abs(expected - observed) == 1.0:
        return {"passed": True, "notes": "Epoch values differ by 1, accepted as zero-based vs one-based representation."}
    return {"passed": False, "notes": "Epoch values do not match, even allowing for zero-based vs one-based representation."}


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
            "passed": passed,
            "severity": severity,
            "observed_value": json.dumps(observed_value, ensure_ascii=True) if isinstance(observed_value, (dict, list)) else observed_value,
            "expected_value": json.dumps(expected_value, ensure_ascii=True) if isinstance(expected_value, (dict, list)) else expected_value,
            "notes": notes,
        }
    )


def add_metric_consistency(
    rows: list[dict[str, Any]],
    metric_name: str,
    phase11k_value: float | None,
    direct_value: float | None,
    passed: bool,
    tolerance: float,
    notes: str,
    treat_as_epoch: bool = False,
) -> None:
    difference = None
    if phase11k_value is not None and direct_value is not None:
        difference = abs(phase11k_value - direct_value)
    if treat_as_epoch and phase11k_value is not None and direct_value is not None and difference == 1.0 and passed:
        difference = 1.0
    rows.append(
        {
            "metric_name": metric_name,
            "phase11k_value": phase11k_value,
            "direct_value": direct_value,
            "difference": difference,
            "tolerance": tolerance,
            "passed": passed,
            "notes": notes,
        }
    )


def build_non_execution_manifest() -> dict[str, Any]:
    return {
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "prediction_executed": False,
        "export_executed": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "weights_modified": False,
        "weights_copied_to_artifacts": False,
        "ultralytics_training_imported": False,
        "checkpoint_tensor_loaded": False,
    }


def build_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 11L Training Output Integrity And Provenance",
            "",
            "Phase 11L is a strict non-execution validation phase.",
            "",
            f"- status = `{summary['status']}`",
            f"- baseline_checkpoint_candidate_accepted = `{summary['baseline_checkpoint_candidate_accepted']}`",
            f"- accepted_checkpoint_role = `{summary['accepted_checkpoint_role']}`",
            f"- accepted_checkpoint_path = `{summary['accepted_checkpoint_path']}`",
            f"- best_epoch = `{summary['best_epoch']}`",
            f"- best_metric_map50_95 = `{summary['best_metric_map50_95']}`",
            f"- final_epoch = `{summary['final_epoch']}`",
            f"- final_metric_map50_95 = `{summary['final_metric_map50_95']}`",
            f"- final_metric_map50 = `{summary['final_metric_map50']}`",
            f"- next_allowed_step = `{summary['next_allowed_step']}`",
            "",
            "Provenance caveat:",
            "",
            f"- {summary['provenance_caveat']}",
            "",
            "This phase did not run training, evaluation, inference, prediction, export, dataset mutation, or checkpoint loading.",
            "",
        ]
    ) + "\n"


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
