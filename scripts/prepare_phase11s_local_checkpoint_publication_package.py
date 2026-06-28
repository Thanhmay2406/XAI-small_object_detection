"""Prepare the Phase 11S local checkpoint publication package without executing publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11S"
DEFAULT_PHASE11R1_SUMMARY = (
    "artifacts/phase11r1_checkpoint_publication_decision_validation/"
    "phase11r1_publication_decision_validation_summary.json"
)
DEFAULT_INVENTORY_CSV = (
    "artifacts/phase11r_checkpoint_publication_decision_gate/phase11r_checkpoint_inventory.csv"
)
DEFAULT_DECISION_CSV = (
    "artifacts/phase11r_checkpoint_publication_decision_gate/phase11r_publication_decision_template.csv"
)
DEFAULT_OUTPUT_DIR = "artifacts/phase11s_local_checkpoint_publication_package"
SUMMARY_NAME = "phase11s_publication_package_summary.json"
CHECKSUMS_NAME = "phase11s_checkpoint_checksums.csv"
CHECKS_NAME = "phase11s_preparation_checks.csv"
MANIFEST_NAME = "phase11s_release_manifest.json"
MODEL_CARD_NAME = "phase11s_model_card_draft.md"
CAVEATS_NAME = "phase11s_publication_caveats.md"
NON_EXECUTION_NAME = "phase11s_non_execution_manifest.json"
README_NAME = "README.md"
STATUS_SUCCESS_METADATA_ONLY = "phase11s_local_checkpoint_publication_package_prepared_metadata_only"
STATUS_SUCCESS_WITH_BINARY = "phase11s_local_checkpoint_publication_package_prepared_with_checkpoint_binary"
STATUS_BLOCKED = "phase11s_local_checkpoint_publication_package_blocked"
NEXT_SUCCESS = "phase11t_manual_checkpoint_publication_execution_gate_or_hold"
NEXT_BLOCKED = "inspect_phase11r1_gate_and_publication_inputs_before_phase11s"
HASH_CHUNK_SIZE = 1024 * 1024
REQUIRED_PHASE11R1_KEYS = {
    "phase",
    "status",
    "decision_validated",
    "phase11r_summary_path",
    "phase11r_inventory_csv_path",
    "phase11r_decision_csv_path",
    "best_checkpoint_path",
    "selected_publication_method",
    "selected_publication_target",
    "reviewer_decision",
    "reviewer_name_or_id",
    "phase11p_metric_caveat_carried_forward",
    "checkpoint_package_preparation_allowed",
    "checkpoint_publication_allowed",
    "checkpoint_upload_executed",
    "checkpoint_load_executed",
    "training_executed",
    "evaluation_executed",
    "inference_executed",
    "dataset_mutated",
    "next_allowed_step",
}
REQUIRED_INVENTORY_FIELDS = {
    "checkpoint_role",
    "path",
    "exists",
    "size_bytes",
    "modified_time_utc",
    "candidate_for_publication",
    "reason",
    "blocked_reason",
}
REQUIRED_DECISION_FIELDS = {
    "phase",
    "checkpoint_path",
    "publication_method",
    "publication_target",
    "include_checksum",
    "include_training_summary_reference",
    "include_metric_caveat_reference",
    "reviewer_decision",
    "reviewer_name_or_id",
    "reviewer_notes",
}
CHECK_FIELDS = ["check_name", "observed_value", "expected_value", "passed", "blocking", "notes"]
CHECKSUM_FIELDS = [
    "checkpoint_role",
    "checkpoint_path",
    "file_exists",
    "size_bytes",
    "checksum_requested",
    "checksum_generated",
    "sha256",
    "notes",
]
OPTIONAL_SUMMARY_PATHS = {
    "phase11j0_summary": (
        "artifacts/phase11j_approved_kaggle_training_command_lock/phase11j0_training_command_lock_summary.json"
    ),
    "phase11n_summary": (
        "artifacts/phase11n_test_evaluation_output_collection_and_validation/phase11n_test_evaluation_output_summary.json"
    ),
    "phase11o_summary": (
        "artifacts/phase11o_manual_test_metric_review/phase11o_manual_metric_review_summary.json"
    ),
    "phase11p_summary": (
        "artifacts/phase11p_final_test_evaluation_report/phase11p_final_test_evaluation_summary.json"
    ),
}
EXPLICIT_BINARY_FIELD = "include_checkpoint_binary_in_local_package"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Phase 11S local checkpoint publication package without upload or model loading."
    )
    parser.add_argument("--phase11r1-summary", default=DEFAULT_PHASE11R1_SUMMARY)
    parser.add_argument("--inventory-csv", default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--decision-csv", default=DEFAULT_DECISION_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_phase11s_local_checkpoint_publication_package(
        phase11r1_summary_path=resolve_repo_path(args.phase11r1_summary),
        inventory_csv_path=resolve_repo_path(args.inventory_csv),
        decision_csv_path=resolve_repo_path(args.decision_csv),
        output_dir=resolve_repo_path(args.output_dir),
    )
    concise = {
        "status": summary["status"],
        "phase11r1_validated": summary["phase11r1_validated"],
        "checkpoint_package_preparation_allowed": summary["checkpoint_package_preparation_allowed"],
        "checkpoint_publication_allowed": summary["checkpoint_publication_allowed"],
        "checkpoint_binary_copied": summary["checkpoint_binary_copied"],
        "checksum_generated": summary["checksum_generated"],
        "next_allowed_step": summary["next_allowed_step"],
    }
    print(json.dumps(concise, indent=2))


def prepare_phase11s_local_checkpoint_publication_package(
    phase11r1_summary_path: Path,
    inventory_csv_path: Path,
    decision_csv_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    phase11r1_summary = read_json_required(phase11r1_summary_path, "Phase 11R.1 summary")
    validate_phase11r1_gate(phase11r1_summary, phase11r1_summary_path, checks)

    inventory_rows = read_csv_required(inventory_csv_path, "Phase 11R inventory CSV")
    validate_inventory(inventory_rows, inventory_csv_path, phase11r1_summary, checks)

    decision_row = read_decision_row(decision_csv_path, checks)
    normalized_decision = normalize_decision_row(decision_row)

    add_check(
        checks,
        "decision_csv_matches_phase11r1_reference",
        str(decision_csv_path.resolve()) == str(resolve_repo_path(phase11r1_summary["phase11r_decision_csv_path"])),
        "error",
        str(decision_csv_path.resolve()),
        str(resolve_repo_path(phase11r1_summary["phase11r_decision_csv_path"])),
        "Phase 11S must use the same decision CSV Phase 11R.1 validated.",
    )
    add_check(
        checks,
        "decision_reviewer_decision_prepare_only",
        normalized_decision.get("reviewer_decision", "") == "approve_publication_package_preparation",
        "error",
        normalized_decision.get("reviewer_decision", ""),
        "approve_publication_package_preparation",
        "Phase 11S only proceeds from the validated local package preparation decision.",
    )
    add_check(
        checks,
        "decision_training_summary_reference_true",
        normalized_decision.get("include_training_summary_reference", "") == "true",
        "error",
        normalized_decision.get("include_training_summary_reference", ""),
        "true",
        "Phase 11S keeps the training summary reference in the local package metadata.",
    )
    add_check(
        checks,
        "decision_metric_caveat_reference_true",
        normalized_decision.get("include_metric_caveat_reference", "") == "true",
        "error",
        normalized_decision.get("include_metric_caveat_reference", ""),
        "true",
        "Phase 11S keeps the metric caveat reference in the local package metadata.",
    )

    selected_checkpoint_path = normalize_path(normalized_decision["checkpoint_path"])
    selected_inventory_row = find_inventory_row(inventory_rows, selected_checkpoint_path)
    add_check(
        checks,
        "selected_checkpoint_in_inventory",
        selected_inventory_row is not None,
        "error",
        selected_checkpoint_path,
        "checkpoint path listed in Phase 11R inventory",
        "Phase 11S packages only a checkpoint candidate already validated by earlier phases.",
    )

    checkpoint_path = resolve_string_path(selected_checkpoint_path)
    checkpoint_exists = checkpoint_path is not None and checkpoint_path.exists() and checkpoint_path.is_file()
    checkpoint_size_bytes = checkpoint_path.stat().st_size if checkpoint_exists else 0
    add_check(
        checks,
        "selected_checkpoint_exists",
        checkpoint_exists,
        "error",
        checkpoint_size_bytes if checkpoint_exists else 0,
        "> 0 byte checkpoint file",
        "Phase 11S requires the selected checkpoint file to exist as a local file.",
    )

    checksum_requested = normalized_decision["include_checksum"] == "true"
    checksum_value = ""
    checksum_generated = False
    if checksum_requested and checkpoint_exists and checkpoint_path is not None:
        checksum_value = compute_file_sha256(checkpoint_path)
        checksum_generated = True

    include_binary_requested = normalized_decision.get(EXPLICIT_BINARY_FIELD, "false") == "true"
    checkpoint_binary_copied = False
    packaged_checkpoint_path = ""
    if include_binary_requested and checkpoint_exists and checkpoint_path is not None:
        packaged_checkpoint_path = str(output_dir / checkpoint_path.name)
        copy_file_streaming(checkpoint_path, Path(packaged_checkpoint_path))
        checkpoint_binary_copied = True

    add_check(
        checks,
        "checkpoint_publication_remains_blocked",
        phase11r1_summary.get("checkpoint_publication_allowed") is False,
        "error",
        phase11r1_summary.get("checkpoint_publication_allowed"),
        False,
        "Phase 11S may prepare a local package only while checkpoint publication stays blocked.",
    )
    add_check(
        checks,
        "checkpoint_upload_not_executed",
        phase11r1_summary.get("checkpoint_upload_executed") is False,
        "error",
        phase11r1_summary.get("checkpoint_upload_executed"),
        False,
        "Phase 11S must begin from a no-upload state.",
    )
    add_check(
        checks,
        "checkpoint_load_not_executed",
        phase11r1_summary.get("checkpoint_load_executed") is False,
        "error",
        phase11r1_summary.get("checkpoint_load_executed"),
        False,
        "Phase 11S must not inherit any checkpoint loading behavior from an earlier phase.",
    )

    phase11r_summary_path = resolve_string_path(str(phase11r1_summary.get("phase11r_summary_path", "")))
    phase11r_summary = read_json_if_exists(phase11r_summary_path)
    optional_references = collect_optional_references(phase11r_summary, warnings)

    reference_warning_checks(optional_references, checks)

    checksum_rows = [
        {
            "checkpoint_role": selected_inventory_row.get("checkpoint_role", "") if selected_inventory_row else "",
            "checkpoint_path": selected_checkpoint_path,
            "file_exists": stringify_csv_value(checkpoint_exists),
            "size_bytes": stringify_csv_value(checkpoint_size_bytes),
            "checksum_requested": stringify_csv_value(checksum_requested),
            "checksum_generated": stringify_csv_value(checksum_generated),
            "sha256": checksum_value,
            "notes": (
                "SHA256 computed by streaming raw bytes only."
                if checksum_generated
                else "Checksum not requested or checkpoint unavailable."
            ),
        }
    ]

    all_blocking_passed = all(row["passed"] == "true" for row in checks if row["blocking"] == "true")
    release_manifest_created = all_blocking_passed
    model_card_draft_created = all_blocking_passed

    status = STATUS_BLOCKED
    next_allowed_step = NEXT_BLOCKED
    if all_blocking_passed:
        status = (
            STATUS_SUCCESS_WITH_BINARY if checkpoint_binary_copied else STATUS_SUCCESS_METADATA_ONLY
        )
        next_allowed_step = NEXT_SUCCESS

    summary = build_summary(
        status=status,
        next_allowed_step=next_allowed_step,
        output_dir=output_dir,
        phase11r1_summary_path=phase11r1_summary_path,
        inventory_csv_path=inventory_csv_path,
        decision_csv_path=decision_csv_path,
        checkpoint_path=selected_checkpoint_path,
        checkpoint_role=selected_inventory_row.get("checkpoint_role", "") if selected_inventory_row else "",
        checkpoint_exists=checkpoint_exists,
        checkpoint_size_bytes=checkpoint_size_bytes,
        checksum_requested=checksum_requested,
        checksum_generated=checksum_generated,
        checksum_value=checksum_value,
        checkpoint_binary_copy_requested=include_binary_requested,
        checkpoint_binary_copied=checkpoint_binary_copied,
        packaged_checkpoint_path=packaged_checkpoint_path,
        phase11r1_summary=phase11r1_summary,
        normalized_decision=normalized_decision,
        optional_references=optional_references,
        warnings=warnings,
        checks=checks,
        release_manifest_created=release_manifest_created,
        model_card_draft_created=model_card_draft_created,
    )

    release_manifest = build_release_manifest(summary)
    publication_caveats = build_publication_caveats(summary, optional_references)
    model_card = build_model_card(summary, optional_references)
    non_execution_manifest = build_non_execution_manifest(summary)

    write_csv(output_dir / CHECKS_NAME, checks, CHECK_FIELDS)
    write_csv(output_dir / CHECKSUMS_NAME, checksum_rows, CHECKSUM_FIELDS)
    write_json(output_dir / SUMMARY_NAME, summary)
    write_json(output_dir / MANIFEST_NAME, release_manifest)
    (output_dir / MODEL_CARD_NAME).write_text(model_card, encoding="utf-8")
    (output_dir / CAVEATS_NAME).write_text(publication_caveats, encoding="utf-8")
    write_json(output_dir / NON_EXECUTION_NAME, non_execution_manifest)
    (output_dir / README_NAME).write_text(build_readme(summary), encoding="utf-8")
    return summary


def validate_phase11r1_gate(
    phase11r1_summary: dict[str, Any],
    phase11r1_summary_path: Path,
    checks: list[dict[str, Any]],
) -> None:
    missing_keys = sorted(REQUIRED_PHASE11R1_KEYS.difference(phase11r1_summary))
    add_check(
        checks,
        "phase11r1_summary_required_keys_present",
        not missing_keys,
        "error",
        missing_keys,
        [],
        "Phase 11S requires the expected Phase 11R.1 fields.",
    )
    if missing_keys:
        raise SystemExit(
            f"Phase 11R.1 summary is missing required keys at {phase11r1_summary_path}: {', '.join(missing_keys)}"
        )

    add_check(
        checks,
        "phase11r1_summary_exists",
        True,
        "info",
        str(phase11r1_summary_path),
        "existing Phase 11R.1 summary JSON",
        "Phase 11S starts from the validated Phase 11R.1 decision summary.",
    )
    add_check(
        checks,
        "phase11r1_status_validated_for_package_preparation",
        phase11r1_summary.get("status")
        == "phase11r1_checkpoint_publication_decision_validated_for_package_preparation",
        "error",
        phase11r1_summary.get("status"),
        "phase11r1_checkpoint_publication_decision_validated_for_package_preparation",
        "Phase 11S requires the exact validated Phase 11R.1 status.",
    )
    add_check(
        checks,
        "phase11r1_decision_validated_true",
        phase11r1_summary.get("decision_validated") is True,
        "error",
        phase11r1_summary.get("decision_validated"),
        True,
        "Phase 11S requires a valid human decision from Phase 11R.1.",
    )
    add_check(
        checks,
        "phase11r1_checkpoint_package_preparation_allowed_true",
        phase11r1_summary.get("checkpoint_package_preparation_allowed") is True,
        "error",
        phase11r1_summary.get("checkpoint_package_preparation_allowed"),
        True,
        "Phase 11S requires explicit package-preparation permission.",
    )
    add_check(
        checks,
        "phase11r1_checkpoint_publication_allowed_false",
        phase11r1_summary.get("checkpoint_publication_allowed") is False,
        "error",
        phase11r1_summary.get("checkpoint_publication_allowed"),
        False,
        "Phase 11S must remain prepare-only and cannot convert the gate into publication approval.",
    )


def validate_inventory(
    inventory_rows: list[dict[str, str]],
    inventory_csv_path: Path,
    phase11r1_summary: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    add_check(
        checks,
        "inventory_has_rows",
        bool(inventory_rows),
        "error",
        len(inventory_rows),
        "at least 1 row",
        "Phase 11S expects the Phase 11R checkpoint inventory.",
    )
    if not inventory_rows:
        raise SystemExit(f"Phase 11R inventory CSV has no rows at {inventory_csv_path}")

    header_ok = REQUIRED_INVENTORY_FIELDS.issubset(inventory_rows[0].keys())
    add_check(
        checks,
        "inventory_has_required_columns",
        header_ok,
        "error",
        sorted(inventory_rows[0].keys()),
        sorted(REQUIRED_INVENTORY_FIELDS),
        "Phase 11S expects the preserved Phase 11R inventory structure.",
    )
    if not header_ok:
        raise SystemExit(f"Phase 11R inventory CSV is missing required columns at {inventory_csv_path}")

    add_check(
        checks,
        "inventory_matches_phase11r1_reference",
        str(inventory_csv_path.resolve()) == str(resolve_repo_path(phase11r1_summary["phase11r_inventory_csv_path"])),
        "error",
        str(inventory_csv_path.resolve()),
        str(resolve_repo_path(phase11r1_summary["phase11r_inventory_csv_path"])),
        "Phase 11S must use the same inventory file Phase 11R.1 validated.",
    )


def read_decision_row(decision_csv_path: Path, checks: list[dict[str, Any]]) -> dict[str, str]:
    rows = read_csv_required(decision_csv_path, "Phase 11R decision CSV")
    add_check(
        checks,
        "decision_csv_single_row",
        len(rows) == 1,
        "error",
        len(rows),
        1,
        "Phase 11S expects exactly one Phase 11R decision row.",
    )
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one decision row in {decision_csv_path}, found {len(rows)}")

    header_ok = REQUIRED_DECISION_FIELDS.issubset(rows[0].keys())
    add_check(
        checks,
        "decision_csv_has_required_columns",
        header_ok,
        "error",
        sorted(rows[0].keys()),
        sorted(REQUIRED_DECISION_FIELDS),
        "Phase 11S expects the validated Phase 11R decision columns.",
    )
    if not header_ok:
        raise SystemExit(f"Decision CSV is missing required columns at {decision_csv_path}")
    return rows[0]


def normalize_decision_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {key: (value or "").strip() for key, value in row.items()}
    for key in [
        "phase",
        "publication_method",
        "publication_target",
        "include_checksum",
        "include_training_summary_reference",
        "include_metric_caveat_reference",
        "reviewer_decision",
        EXPLICIT_BINARY_FIELD,
    ]:
        if key in normalized:
            normalized[key] = normalized[key].strip().lower()
    return normalized


def find_inventory_row(inventory_rows: list[dict[str, str]], checkpoint_path: str) -> dict[str, str] | None:
    for row in inventory_rows:
        if normalize_path(row.get("path", "")) == checkpoint_path:
            return row
    return None


def collect_optional_references(
    phase11r_summary: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}

    for label, relative_path in OPTIONAL_SUMMARY_PATHS.items():
        path = resolve_repo_path(relative_path)
        payload = read_json_if_exists(path)
        available = payload is not None
        references[label] = {
            "path": str(path),
            "available": available,
            "status": str(payload.get("status", "")) if payload else "",
        }
        if payload:
            references[label]["phase"] = str(payload.get("phase", ""))
        else:
            warnings.append(f"Optional reference missing: {label} at {path}")

    phase11p_report_path = ""
    if phase11r_summary:
        phase11p_report_path = str(phase11r_summary.get("phase11p_report_path", "")).strip()
    report_path = resolve_string_path(phase11p_report_path)
    report_available = report_path is not None and report_path.exists() and report_path.is_file()
    references["phase11p_report"] = {
        "path": str(report_path) if report_path else phase11p_report_path,
        "available": report_available,
        "status": "present" if report_available else "",
    }
    if not report_available:
        warnings.append(
            "Optional reference missing: phase11p_report_path carried forward from Phase 11R is unavailable."
        )
    return references


def reference_warning_checks(
    optional_references: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    for label, metadata in optional_references.items():
        add_check(
            checks,
            f"{label}_available",
            metadata["available"],
            "warning",
            metadata["path"],
            "optional reference available",
            "Missing optional references are recorded as warnings and do not block Phase 11S.",
        )


def build_summary(
    *,
    status: str,
    next_allowed_step: str,
    output_dir: Path,
    phase11r1_summary_path: Path,
    inventory_csv_path: Path,
    decision_csv_path: Path,
    checkpoint_path: str,
    checkpoint_role: str,
    checkpoint_exists: bool,
    checkpoint_size_bytes: int,
    checksum_requested: bool,
    checksum_generated: bool,
    checksum_value: str,
    checkpoint_binary_copy_requested: bool,
    checkpoint_binary_copied: bool,
    packaged_checkpoint_path: str,
    phase11r1_summary: dict[str, Any],
    normalized_decision: dict[str, str],
    optional_references: dict[str, dict[str, Any]],
    warnings: list[str],
    checks: list[dict[str, Any]],
    release_manifest_created: bool,
    model_card_draft_created: bool,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": status,
        "phase11r1_summary_path": str(phase11r1_summary_path),
        "phase11r1_validated": phase11r1_summary.get("decision_validated") is True,
        "phase11r1_status": phase11r1_summary.get("status", ""),
        "phase11r_inventory_csv_path": str(inventory_csv_path),
        "phase11r_decision_csv_path": str(decision_csv_path),
        "checkpoint_role": checkpoint_role,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "publication_method": normalized_decision.get("publication_method", ""),
        "publication_target": normalized_decision.get("publication_target", ""),
        "reviewer_decision": normalized_decision.get("reviewer_decision", ""),
        "reviewer_name_or_id": normalized_decision.get("reviewer_name_or_id", ""),
        "reviewer_notes": normalized_decision.get("reviewer_notes", ""),
        "include_checksum": normalized_decision.get("include_checksum", ""),
        "include_training_summary_reference": normalized_decision.get(
            "include_training_summary_reference", ""
        ),
        "include_metric_caveat_reference": normalized_decision.get(
            "include_metric_caveat_reference", ""
        ),
        "include_checkpoint_binary_in_local_package": normalized_decision.get(
            EXPLICIT_BINARY_FIELD, "false"
        ),
        "checkpoint_package_preparation_allowed": phase11r1_summary.get(
            "checkpoint_package_preparation_allowed"
        )
        is True,
        "checkpoint_publication_allowed": False,
        "checkpoint_upload_executed": False,
        "checkpoint_load_executed": False,
        "checkpoint_binary_copy_requested": checkpoint_binary_copy_requested,
        "checkpoint_binary_copied": checkpoint_binary_copied,
        "packaged_checkpoint_path": packaged_checkpoint_path,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "checksum_generated": checksum_generated,
        "checksum_sha256": checksum_value,
        "model_card_draft_created": model_card_draft_created,
        "release_manifest_created": release_manifest_created,
        "publication_caveats_created": True,
        "optional_reference_count": len(optional_references),
        "optional_reference_warning_count": len(warnings),
        "optional_reference_warnings": warnings,
        "optional_references": optional_references,
        "phase11p_metric_caveat_carried_forward": phase11r1_summary.get(
            "phase11p_metric_caveat_carried_forward", ""
        ),
        "checks_csv_path": str(output_dir / CHECKS_NAME),
        "checksums_csv_path": str(output_dir / CHECKSUMS_NAME),
        "release_manifest_path": str(output_dir / MANIFEST_NAME),
        "model_card_draft_path": str(output_dir / MODEL_CARD_NAME),
        "publication_caveats_path": str(output_dir / CAVEATS_NAME),
        "non_execution_manifest_path": str(output_dir / NON_EXECUTION_NAME),
        "readme_path": str(output_dir / README_NAME),
        "failed_check_count": sum(1 for row in checks if row["passed"] == "false"),
        "blocking_check_failure_count": sum(
            1 for row in checks if row["passed"] == "false" and row["blocking"] == "true"
        ),
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_release_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": summary["status"],
        "package_mode": "checkpoint_binary_included" if summary["checkpoint_binary_copied"] else "metadata_only",
        "checkpoint_path": summary["checkpoint_path"],
        "packaged_checkpoint_path": summary["packaged_checkpoint_path"],
        "checkpoint_size_bytes": summary["checkpoint_size_bytes"],
        "checkpoint_sha256": summary["checksum_sha256"],
        "checkpoint_sha256_generated": summary["checksum_generated"],
        "publication_method": summary["publication_method"],
        "publication_target": summary["publication_target"],
        "reviewer_decision": summary["reviewer_decision"],
        "reviewer_name_or_id": summary["reviewer_name_or_id"],
        "checkpoint_package_preparation_allowed": summary["checkpoint_package_preparation_allowed"],
        "checkpoint_publication_allowed": False,
        "checkpoint_upload_executed": False,
        "checkpoint_load_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "references": summary["optional_references"],
        "phase11p_metric_caveat_carried_forward": summary["phase11p_metric_caveat_carried_forward"],
        "next_allowed_step": summary["next_allowed_step"],
    }


def build_publication_caveats(
    summary: dict[str, Any],
    optional_references: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Phase 11S Publication Caveats",
        "",
        "This package is prepared locally for metadata review only.",
        "",
        "Core restrictions:",
        "",
        "- Checkpoint publication remains blocked.",
        "- No checkpoint upload has been executed.",
        "- No checkpoint loading with torch, ultralytics, YOLO, or any model library has been performed.",
        "- No training, evaluation, inference, or dataset mutation has been performed.",
        "",
        "Carried-forward metric provenance caveat:",
        "",
        summary["phase11p_metric_caveat_carried_forward"] or "No Phase 11P metric caveat text was available.",
        "",
        "Optional reference availability:",
        "",
    ]
    for label, metadata in optional_references.items():
        availability = "available" if metadata["available"] else "missing"
        lines.append(f"- {label}: {availability} | {metadata['path']}")
    if summary["optional_reference_warnings"]:
        lines.extend(["", "Warnings:", ""])
        for warning in summary["optional_reference_warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def build_model_card(summary: dict[str, Any], optional_references: dict[str, dict[str, Any]]) -> str:
    phase11p_report_path = optional_references.get("phase11p_report", {}).get("path", "")
    phase11j0_status = optional_references.get("phase11j0_summary", {}).get("status", "")
    phase11n_status = optional_references.get("phase11n_summary", {}).get("status", "")
    phase11o_status = optional_references.get("phase11o_summary", {}).get("status", "")
    phase11p_status = optional_references.get("phase11p_summary", {}).get("status", "")

    return f"""# Phase 11S Model Card Draft

## Model Identity

- Phase: `{PHASE}`
- Checkpoint role: `{summary["checkpoint_role"]}`
- Checkpoint path: `{summary["checkpoint_path"]}`
- Publication method: `{summary["publication_method"]}`
- Publication target: `{summary["publication_target"]}`
- Reviewer decision: `{summary["reviewer_decision"]}`

## Package Scope

This draft was prepared as a local checkpoint publication package only.
Checkpoint publication remains blocked pending a later manual execution gate.

- checkpoint_package_preparation_allowed = `{summary["checkpoint_package_preparation_allowed"]}`
- checkpoint_publication_allowed = `{summary["checkpoint_publication_allowed"]}`
- checkpoint_upload_executed = `{summary["checkpoint_upload_executed"]}`
- checkpoint_load_executed = `{summary["checkpoint_load_executed"]}`
- checkpoint_binary_copied = `{summary["checkpoint_binary_copied"]}`
- checksum_generated = `{summary["checksum_generated"]}`

## Provenance

- Phase 11J.0 status: `{phase11j0_status or "missing optional reference"}`
- Phase 11N status: `{phase11n_status or "missing optional reference"}`
- Phase 11O status: `{phase11o_status or "missing optional reference"}`
- Phase 11P status: `{phase11p_status or "missing optional reference"}`
- Phase 11P report path: `{phase11p_report_path or "missing optional reference"}`

## Metric Caveat

{summary["phase11p_metric_caveat_carried_forward"] or "No carried-forward metric caveat text was available."}

## Usage Restrictions

- Do not treat this draft as publication approval.
- Do not upload or publish the checkpoint from Phase 11S alone.
- Do not claim validated test metrics beyond the carried-forward caveated references.
- Do not mutate the dataset or rerun training/evaluation/inference from this phase.

## Next Step

`{summary["next_allowed_step"]}`
"""


def build_non_execution_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": summary["status"],
        "validation_only": False,
        "package_preparation_only": True,
        "checkpoint_upload_executed": False,
        "checkpoint_load_executed": False,
        "checkpoint_binary_copied": summary["checkpoint_binary_copied"],
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "prediction_executed": False,
        "export_executed": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "weights_modified": False,
        "git_add_executed": False,
        "git_commit_executed": False,
        "git_push_executed": False,
        "notes": (
            "Phase 11S prepares a local publication package only. "
            "It never loads the checkpoint with model libraries, never uploads it, "
            "and never runs training, evaluation, or inference."
        ),
    }


def build_readme(summary: dict[str, Any]) -> str:
    return f"""# Phase 11S Local Checkpoint Publication Package

This artifact bundle prepares a local checkpoint publication package without executing publication.

- status = `{summary["status"]}`
- phase11r1_validated = `{summary["phase11r1_validated"]}`
- checkpoint_package_preparation_allowed = `{summary["checkpoint_package_preparation_allowed"]}`
- checkpoint_publication_allowed = `{summary["checkpoint_publication_allowed"]}`
- checkpoint_upload_executed = `{summary["checkpoint_upload_executed"]}`
- checkpoint_load_executed = `{summary["checkpoint_load_executed"]}`
- checkpoint_binary_copied = `{summary["checkpoint_binary_copied"]}`
- checksum_generated = `{summary["checksum_generated"]}`
- next_allowed_step = `{summary["next_allowed_step"]}`

Outputs:

- `phase11s_publication_package_summary.json`
- `phase11s_checkpoint_checksums.csv`
- `phase11s_release_manifest.json`
- `phase11s_model_card_draft.md`
- `phase11s_publication_caveats.md`
- `phase11s_non_execution_manifest.json`
- `README.md`

Phase 11S does not upload the checkpoint, does not load it with a model library, and does not run training, evaluation, inference, or dataset mutation.
"""


def add_check(
    checks: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    severity: str,
    observed_value: Any,
    expected_value: Any,
    notes: str,
) -> None:
    checks.append(
        {
            "check_name": check_name,
            "observed_value": stringify_csv_value(observed_value),
            "expected_value": stringify_csv_value(expected_value),
            "passed": "true" if passed else "false",
            "blocking": "true" if severity == "error" else "false",
            "notes": notes,
        }
    )


def normalize_path(path_str: str) -> str:
    if not path_str.strip():
        return ""
    path = Path(path_str)
    resolved = path if path.is_absolute() else (PROJECT_ROOT / path)
    return str(resolved.resolve())


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_string_path(path_str: str) -> Path | None:
    if not path_str.strip():
        return None
    return resolve_repo_path(path_str)


def read_json_required(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    return read_json(path, label)


def read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return read_json(path, str(path))


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object at {path}")
    return payload


def read_csv_required(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{label} is missing a header row at {path}")
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
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file_streaming(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, HASH_CHUNK_SIZE)


if __name__ == "__main__":
    main()
