"""Validate the Phase 11T manual checkpoint publication execution gate without executing publication."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11T"
DEFAULT_PHASE11S_SUMMARY = (
    "artifacts/phase11s_local_checkpoint_publication_package/phase11s_publication_package_summary.json"
)
DEFAULT_DECISION_CSV = (
    "artifacts/phase11t_manual_checkpoint_publication_execution_gate/"
    "phase11t_publication_execution_decision_template.csv"
)
DEFAULT_OUTPUT_DIR = "artifacts/phase11t_manual_checkpoint_publication_execution_gate"
TEMPLATE_NAME = "phase11t_publication_execution_decision_template.csv"
USED_NAME = "phase11t_publication_execution_decision_used.csv"
SUMMARY_NAME = "phase11t_publication_execution_gate_summary.json"
CHECKS_NAME = "phase11t_publication_execution_checks.csv"
MANIFEST_NAME = "phase11t_non_execution_manifest.json"
README_NAME = "README.md"
STATUS_WAITING = "phase11t_blocked_waiting_human_checkpoint_publication_execution_decision"
STATUS_APPROVED = "phase11t_checkpoint_publication_execution_gate_approved"
STATUS_INVALID = "phase11t_blocked_invalid_checkpoint_publication_execution_decision"
NEXT_WAITING = "fill_phase11t_manual_decision_or_hold"
NEXT_APPROVED = "phase11u_prepare_manual_publication_execution_adapter_or_hold"
NEXT_INVALID = "fill_phase11t_manual_decision_or_hold"
EXPECTED_PHASE11S_STATUS = "phase11s_local_checkpoint_publication_package_prepared_metadata_only"
EXPECTED_PHASE11S_NEXT = "phase11t_manual_checkpoint_publication_execution_gate_or_hold"
REQUIRED_PHASE11S_KEYS = {
    "phase",
    "status",
    "phase11r1_validated",
    "checkpoint_package_preparation_allowed",
    "checkpoint_publication_allowed",
    "checkpoint_upload_executed",
    "checkpoint_load_executed",
    "checkpoint_binary_copied",
    "training_executed",
    "evaluation_executed",
    "inference_executed",
    "dataset_mutated",
    "next_allowed_step",
}
DECISION_FIELDS = [
    "decision_id",
    "human_reviewer",
    "decision",
    "publication_execution_allowed",
    "checkpoint_upload_allowed",
    "checkpoint_binary_publication_allowed",
    "checkpoint_load_allowed",
    "publication_target",
    "notes",
]
CHECK_FIELDS = ["check_name", "passed", "severity", "observed_value", "expected_value", "notes"]
PLACEHOLDER_TOKENS = {
    "",
    "pending",
    "pending_manual_decision",
    "fill manually",
    "manual_reviewer",
    "placeholder",
    "unknown",
    "n/a",
    "na",
    "tbd",
    "todo",
    "example",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a manual checkpoint publication execution decision without execution."
    )
    parser.add_argument("--phase11s-summary", default=DEFAULT_PHASE11S_SUMMARY)
    parser.add_argument("--decision-csv", default=DEFAULT_DECISION_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_phase11t_manual_checkpoint_publication_execution_gate(
        phase11s_summary_path=resolve_repo_path(args.phase11s_summary),
        decision_csv_path=resolve_repo_path(args.decision_csv),
        output_dir=resolve_repo_path(args.output_dir),
    )
    concise = {
        "status": summary["status"],
        "phase11s_validated": summary["phase11s_validated"],
        "publication_execution_allowed": summary["publication_execution_allowed"],
        "checkpoint_upload_allowed": summary["checkpoint_upload_allowed"],
        "checkpoint_load_allowed": summary["checkpoint_load_allowed"],
        "checkpoint_publication_allowed": summary["checkpoint_publication_allowed"],
        "next_allowed_step": summary["next_allowed_step"],
    }
    print(json.dumps(concise, indent=2))


def validate_phase11t_manual_checkpoint_publication_execution_gate(
    phase11s_summary_path: Path,
    decision_csv_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    phase11s_summary = read_json_required(phase11s_summary_path, "Phase 11S summary")
    phase11s_validated = validate_phase11s_summary(phase11s_summary, phase11s_summary_path, checks)

    template_row = build_decision_template_row()
    template_path = output_dir / TEMPLATE_NAME
    write_csv(template_path, [template_row], DECISION_FIELDS)

    decision_result = validate_optional_decision_csv(
        decision_csv_path=decision_csv_path,
        expected_template_path=template_path,
        checks=checks,
    )
    used_rows = decision_result["used_rows"] if decision_result["used_rows"] else [template_row]
    write_csv(output_dir / USED_NAME, used_rows, DECISION_FIELDS)

    summary = build_summary(
        phase11s_summary=phase11s_summary,
        phase11s_summary_path=phase11s_summary_path,
        decision_csv_path=decision_csv_path,
        template_path=template_path,
        output_dir=output_dir,
        checks=checks,
        phase11s_validated=phase11s_validated,
        decision_result=decision_result,
    )
    manifest = build_non_execution_manifest(summary)

    write_csv(output_dir / CHECKS_NAME, checks, CHECK_FIELDS)
    write_json(output_dir / SUMMARY_NAME, summary)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / README_NAME).write_text(build_readme(summary), encoding="utf-8")
    return summary


def validate_phase11s_summary(
    summary: dict[str, Any],
    summary_path: Path,
    checks: list[dict[str, Any]],
) -> bool:
    missing = sorted(REQUIRED_PHASE11S_KEYS.difference(summary))
    add_check(
        checks,
        "phase11s_required_keys_present",
        not missing,
        "error",
        missing,
        [],
        "Phase 11T requires the expected Phase 11S summary fields.",
    )
    if missing:
        raise SystemExit(f"Phase 11S summary is missing required keys at {summary_path}: {', '.join(missing)}")

    add_check(
        checks,
        "phase11s_summary_exists",
        True,
        "info",
        str(summary_path),
        "existing Phase 11S summary JSON",
        "Phase 11T consumes the Phase 11S metadata-only package summary.",
    )
    add_check(
        checks,
        "phase11s_phase_matches",
        summary.get("phase") == "11S",
        "error",
        summary.get("phase"),
        "11S",
        "Phase 11T should read only the Phase 11S summary.",
    )
    add_check(
        checks,
        "phase11s_status_prepare_only",
        summary.get("status") == EXPECTED_PHASE11S_STATUS,
        "warning",
        summary.get("status"),
        EXPECTED_PHASE11S_STATUS,
        "Phase 11T expects the normal metadata-only Phase 11S success state.",
    )
    add_check(
        checks,
        "phase11s_phase11r1_validated",
        summary.get("phase11r1_validated") is True,
        "error",
        summary.get("phase11r1_validated"),
        True,
        "Phase 11T requires Phase 11S to come from a validated Phase 11R.1 gate.",
    )
    add_check(
        checks,
        "phase11s_checkpoint_package_preparation_allowed",
        summary.get("checkpoint_package_preparation_allowed") is True,
        "error",
        summary.get("checkpoint_package_preparation_allowed"),
        True,
        "Phase 11T requires the Phase 11S package-preparation permission.",
    )
    add_check(
        checks,
        "phase11s_checkpoint_publication_allowed_false",
        summary.get("checkpoint_publication_allowed") is False,
        "error",
        summary.get("checkpoint_publication_allowed"),
        False,
        "Phase 11S must still have checkpoint publication blocked before Phase 11T.",
    )
    add_check(
        checks,
        "phase11s_checkpoint_upload_executed_false",
        summary.get("checkpoint_upload_executed") is False,
        "error",
        summary.get("checkpoint_upload_executed"),
        False,
        "Phase 11T requires no prior checkpoint upload execution.",
    )
    add_check(
        checks,
        "phase11s_checkpoint_load_executed_false",
        summary.get("checkpoint_load_executed") is False,
        "error",
        summary.get("checkpoint_load_executed"),
        False,
        "Phase 11T requires no prior checkpoint load execution.",
    )
    add_check(
        checks,
        "phase11s_checkpoint_binary_copied_false",
        summary.get("checkpoint_binary_copied") is False,
        "error",
        summary.get("checkpoint_binary_copied"),
        False,
        "Phase 11T requires the Phase 11S package to remain metadata-only by default.",
    )
    add_check(
        checks,
        "phase11s_no_training_evaluation_inference_dataset_mutation",
        all(
            summary.get(field) is False
            for field in ["training_executed", "evaluation_executed", "inference_executed", "dataset_mutated"]
        ),
        "error",
        {
            "training_executed": summary.get("training_executed"),
            "evaluation_executed": summary.get("evaluation_executed"),
            "inference_executed": summary.get("inference_executed"),
            "dataset_mutated": summary.get("dataset_mutated"),
        },
        {
            "training_executed": False,
            "evaluation_executed": False,
            "inference_executed": False,
            "dataset_mutated": False,
        },
        "Phase 11T requires a fully non-executed upstream package-preparation phase.",
    )
    add_check(
        checks,
        "phase11s_next_allowed_step_matches",
        summary.get("next_allowed_step") == EXPECTED_PHASE11S_NEXT,
        "warning",
        summary.get("next_allowed_step"),
        EXPECTED_PHASE11S_NEXT,
        "The normal Phase 11S handoff points directly to Phase 11T.",
    )

    return all(
        row["passed"] == "true" for row in checks if row["severity"] == "error"
    )


def build_decision_template_row() -> dict[str, str]:
    return {
        "decision_id": "phase11t_manual_publication_decision_001",
        "human_reviewer": "",
        "decision": "pending_manual_decision",
        "publication_execution_allowed": "false",
        "checkpoint_upload_allowed": "false",
        "checkpoint_binary_publication_allowed": "false",
        "checkpoint_load_allowed": "false",
        "publication_target": "",
        "notes": (
            "Fill manually after real human review. Phase 11T is a gate only; "
            "it must not upload, publish, load, or execute the checkpoint."
        ),
    }


def validate_optional_decision_csv(
    decision_csv_path: Path,
    expected_template_path: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provided": False,
        "valid": False,
        "errors": [],
        "normalized_row": None,
        "used_rows": [],
    }

    same_as_template = decision_csv_path.resolve() == expected_template_path.resolve()
    add_check(
        checks,
        "decision_csv_expected_path",
        same_as_template,
        "warning",
        str(decision_csv_path.resolve()),
        str(expected_template_path.resolve()),
        "By default Phase 11T reads the decision template path in its own output directory.",
    )

    if not decision_csv_path.exists():
        add_check(
            checks,
            "decision_csv_exists",
            False,
            "warning",
            str(decision_csv_path),
            "existing filled decision CSV path",
            "No filled decision CSV exists yet, so Phase 11T remains blocked waiting for human input.",
        )
        return result

    rows = read_csv_required(decision_csv_path, "Phase 11T decision CSV")
    result["provided"] = True
    add_check(
        checks,
        "decision_csv_exists",
        True,
        "info",
        str(decision_csv_path),
        "existing filled decision CSV path",
        "Phase 11T found a candidate decision CSV to validate.",
    )
    add_check(
        checks,
        "decision_csv_row_count",
        len(rows) == 1,
        "error",
        len(rows),
        1,
        "Phase 11T accepts exactly one decision row.",
    )
    if len(rows) != 1:
        result["errors"].append(f"Expected exactly one decision row, found {len(rows)}.")
        return result

    row = rows[0]
    header_ok = set(DECISION_FIELDS).issubset(row.keys())
    add_check(
        checks,
        "decision_csv_has_required_columns",
        header_ok,
        "error",
        sorted(row.keys()),
        DECISION_FIELDS,
        "Phase 11T decision CSV must preserve the exact decision columns.",
    )
    if not header_ok:
        result["errors"].append("Decision CSV is missing one or more required columns.")
        return result

    normalized = {field: (row.get(field, "") or "").strip() for field in DECISION_FIELDS}
    lower_fields = {
        "decision",
        "publication_execution_allowed",
        "checkpoint_upload_allowed",
        "checkpoint_binary_publication_allowed",
        "checkpoint_load_allowed",
        "publication_target",
    }
    for field in lower_fields:
        normalized[field] = normalized[field].lower()

    unfilled_template = is_unfilled_decision(normalized)
    add_check(
        checks,
        "decision_csv_filled_by_human",
        not unfilled_template,
        "warning",
        {
            "decision": normalized["decision"],
            "human_reviewer": normalized["human_reviewer"],
            "publication_target": normalized["publication_target"],
        },
        "non-pending decision with filled reviewer and publication_target",
        "An unfilled template keeps Phase 11T blocked waiting for human input rather than invalid.",
    )
    if unfilled_template:
        result["provided"] = False
        result["used_rows"] = [normalized]
        return result

    validate_filled_decision(normalized, checks, result["errors"])
    result["valid"] = not result["errors"]
    result["normalized_row"] = normalized
    result["used_rows"] = [normalized]
    return result


def is_unfilled_decision(normalized: dict[str, str]) -> bool:
    return (
        normalized["decision"] == "pending_manual_decision"
        and not is_non_placeholder(normalized["human_reviewer"])
        and not is_non_placeholder(normalized["publication_target"])
    )


def validate_filled_decision(
    normalized: dict[str, str],
    checks: list[dict[str, Any]],
    errors: list[str],
) -> None:
    decision_id_present = is_non_placeholder(normalized["decision_id"])
    add_check(
        checks,
        "decision_id_present",
        decision_id_present,
        "error",
        normalized["decision_id"],
        "non-empty decision_id",
        "Phase 11T requires an explicit decision identifier for auditability.",
    )
    if not decision_id_present:
        errors.append("decision_id must be filled.")

    reviewer_present = is_non_placeholder(normalized["human_reviewer"])
    add_check(
        checks,
        "human_reviewer_present",
        reviewer_present,
        "error",
        normalized["human_reviewer"],
        "non-empty human_reviewer",
        "Phase 11T requires a real human reviewer identity.",
    )
    if not reviewer_present:
        errors.append("human_reviewer must be filled.")

    decision_value = normalized["decision"]
    decision_allowed = decision_value in {
        "approve_manual_checkpoint_publication_execution",
        "hold",
        "reject",
        "pending_manual_decision",
    }
    add_check(
        checks,
        "decision_allowed",
        decision_allowed,
        "error",
        decision_value,
        "approve_manual_checkpoint_publication_execution, hold, reject, or pending_manual_decision",
        "Phase 11T supports a small set of explicit human gate outcomes.",
    )
    if not decision_allowed:
        errors.append("decision is not allowed.")

    for field in [
        "publication_execution_allowed",
        "checkpoint_upload_allowed",
        "checkpoint_binary_publication_allowed",
        "checkpoint_load_allowed",
    ]:
        bool_ok = normalized[field] in {"true", "false"}
        add_check(
            checks,
            f"{field}_boolean",
            bool_ok,
            "error",
            normalized[field],
            "true or false",
            f"{field} must be explicitly true or false.",
        )
        if not bool_ok:
            errors.append(f"{field} must be true or false.")

    publication_target_present = is_non_placeholder(normalized["publication_target"])
    add_check(
        checks,
        "publication_target_present",
        publication_target_present,
        "error",
        normalized["publication_target"],
        "non-empty publication_target",
        "Phase 11T requires the human reviewer to specify the intended publication target.",
    )
    if not publication_target_present:
        errors.append("publication_target must be filled.")

    notes_present = is_non_placeholder(normalized["notes"])
    add_check(
        checks,
        "notes_present",
        notes_present,
        "warning",
        normalized["notes"],
        "non-empty notes",
        "Reviewer notes are strongly preferred to explain the approval, hold, or rejection context.",
    )

    if decision_value == "pending_manual_decision":
        add_check(
            checks,
            "pending_manual_decision_blocking",
            False,
            "error",
            decision_value,
            "non-pending human decision",
            "pending_manual_decision always keeps Phase 11T blocked.",
        )
        errors.append("decision is still pending_manual_decision.")
        return

    if decision_value == "approve_manual_checkpoint_publication_execution":
        execution_ok = normalized["publication_execution_allowed"] == "true"
        upload_ok = normalized["checkpoint_upload_allowed"] in {"true", "false"}
        binary_ok = normalized["checkpoint_binary_publication_allowed"] == "true"
        load_ok = normalized["checkpoint_load_allowed"] == "false"

        add_check(
            checks,
            "approval_publication_execution_allowed_true",
            execution_ok,
            "error",
            normalized["publication_execution_allowed"],
            "true",
            "Approval requires an explicit publication_execution_allowed=true acknowledgment.",
        )
        add_check(
            checks,
            "approval_checkpoint_binary_publication_allowed_true",
            binary_ok,
            "error",
            normalized["checkpoint_binary_publication_allowed"],
            "true",
            "Approval requires intentionally allowing checkpoint binary publication.",
        )
        add_check(
            checks,
            "approval_checkpoint_load_allowed_false",
            load_ok,
            "error",
            normalized["checkpoint_load_allowed"],
            "false",
            "Phase 11T approval must not allow checkpoint loading.",
        )
        add_check(
            checks,
            "approval_checkpoint_upload_allowed_explicit",
            upload_ok,
            "error",
            normalized["checkpoint_upload_allowed"],
            "true or false",
            "Approval must explicitly state whether a later phase may upload the checkpoint.",
        )

        if not execution_ok:
            errors.append("approve_manual_checkpoint_publication_execution requires publication_execution_allowed=true.")
        if not binary_ok:
            errors.append(
                "approve_manual_checkpoint_publication_execution requires checkpoint_binary_publication_allowed=true."
            )
        if not load_ok:
            errors.append("approve_manual_checkpoint_publication_execution requires checkpoint_load_allowed=false.")

    if decision_value in {"hold", "reject"} and not notes_present:
        errors.append(f"{decision_value} decision requires notes.")


def build_summary(
    *,
    phase11s_summary: dict[str, Any],
    phase11s_summary_path: Path,
    decision_csv_path: Path,
    template_path: Path,
    output_dir: Path,
    checks: list[dict[str, Any]],
    phase11s_validated: bool,
    decision_result: dict[str, Any],
) -> dict[str, Any]:
    normalized = decision_result["normalized_row"]
    decision_value = normalized["decision"] if normalized else ""
    publication_execution_allowed = False
    checkpoint_upload_allowed = False
    checkpoint_binary_publication_allowed = False
    checkpoint_load_allowed = False
    checkpoint_publication_allowed = False
    status = STATUS_WAITING
    next_allowed_step = NEXT_WAITING
    decision_block_reason = ""

    if decision_result["provided"]:
        if decision_result["valid"] and decision_value == "approve_manual_checkpoint_publication_execution":
            publication_execution_allowed = True
            checkpoint_upload_allowed = normalized["checkpoint_upload_allowed"] == "true"
            checkpoint_binary_publication_allowed = (
                normalized["checkpoint_binary_publication_allowed"] == "true"
            )
            checkpoint_load_allowed = normalized["checkpoint_load_allowed"] == "true"
            checkpoint_publication_allowed = True
            status = STATUS_APPROVED
            next_allowed_step = NEXT_APPROVED
        elif decision_result["valid"]:
            status = STATUS_INVALID
            next_allowed_step = NEXT_INVALID
            decision_block_reason = f"decision={decision_value} does not approve publication execution."
        else:
            status = STATUS_INVALID
            next_allowed_step = NEXT_INVALID
            decision_block_reason = "; ".join(decision_result["errors"])

    return {
        "phase": PHASE,
        "status": status,
        "phase11s_summary_path": str(phase11s_summary_path),
        "phase11s_validated": phase11s_validated,
        "phase11s_status": phase11s_summary.get("status", ""),
        "phase11s_phase11r1_validated": phase11s_summary.get("phase11r1_validated"),
        "phase11s_checkpoint_package_preparation_allowed": phase11s_summary.get(
            "checkpoint_package_preparation_allowed"
        ),
        "phase11s_checkpoint_publication_allowed": phase11s_summary.get(
            "checkpoint_publication_allowed"
        ),
        "phase11s_checkpoint_upload_executed": phase11s_summary.get("checkpoint_upload_executed"),
        "phase11s_checkpoint_load_executed": phase11s_summary.get("checkpoint_load_executed"),
        "decision_csv_path": str(decision_csv_path),
        "decision_template_csv_path": str(template_path),
        "decision_csv_provided": decision_result["provided"],
        "decision_validated": decision_result["valid"],
        "decision_id": normalized["decision_id"] if normalized else "",
        "human_reviewer": normalized["human_reviewer"] if normalized else "",
        "decision": decision_value,
        "publication_target": normalized["publication_target"] if normalized else "",
        "notes": normalized["notes"] if normalized else "",
        "decision_block_reason": decision_block_reason,
        "publication_execution_allowed": publication_execution_allowed,
        "checkpoint_upload_allowed": checkpoint_upload_allowed,
        "checkpoint_load_allowed": checkpoint_load_allowed,
        "checkpoint_binary_publication_allowed": checkpoint_binary_publication_allowed,
        "checkpoint_publication_allowed": checkpoint_publication_allowed,
        "checkpoint_upload_executed": False,
        "checkpoint_load_executed": False,
        "checkpoint_binary_copied": False,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutation_executed": False,
        "validation_error_count": len(decision_result["errors"]),
        "validation_errors": decision_result["errors"],
        "checks_csv_path": str(output_dir / CHECKS_NAME),
        "summary_json_path": str(output_dir / SUMMARY_NAME),
        "non_execution_manifest_path": str(output_dir / MANIFEST_NAME),
        "readme_path": str(output_dir / README_NAME),
        "failed_check_count": sum(1 for row in checks if row["passed"] == "false"),
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_non_execution_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": summary["status"],
        "decision_gate_only": True,
        "checkpoint_upload_executed": False,
        "checkpoint_load_executed": False,
        "checkpoint_binary_copied": False,
        "checkpoint_publication_remote_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutation_executed": False,
        "notes": (
            "Phase 11T validates only a manual checkpoint publication execution decision. "
            "It does not upload, publish, load, copy checkpoint binaries, train, evaluate, infer, or mutate data."
        ),
    }


def build_readme(summary: dict[str, Any]) -> str:
    return f"""# Phase 11T Manual Checkpoint Publication Execution Gate

This artifact bundle validates whether a real human has explicitly approved a later checkpoint publication execution phase.

- status = `{summary["status"]}`
- phase11s_validated = `{summary["phase11s_validated"]}`
- publication_execution_allowed = `{summary["publication_execution_allowed"]}`
- checkpoint_upload_allowed = `{summary["checkpoint_upload_allowed"]}`
- checkpoint_load_allowed = `{summary["checkpoint_load_allowed"]}`
- checkpoint_binary_publication_allowed = `{summary["checkpoint_binary_publication_allowed"]}`
- checkpoint_publication_allowed = `{summary["checkpoint_publication_allowed"]}`
- checkpoint_upload_executed = `{summary["checkpoint_upload_executed"]}`
- checkpoint_load_executed = `{summary["checkpoint_load_executed"]}`
- checkpoint_binary_copied = `{summary["checkpoint_binary_copied"]}`
- next_allowed_step = `{summary["next_allowed_step"]}`

Outputs:

- `phase11t_publication_execution_decision_template.csv`
- `phase11t_publication_execution_decision_used.csv`
- `phase11t_publication_execution_gate_summary.json`
- `phase11t_publication_execution_checks.csv`
- `phase11t_non_execution_manifest.json`
- `README.md`

Phase 11T never uploads, publishes, loads, copies, trains, evaluates, infers, or mutates the dataset.
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
            "passed": "true" if passed else "false",
            "severity": severity,
            "observed_value": stringify_csv_value(observed_value),
            "expected_value": stringify_csv_value(expected_value),
            "notes": notes,
        }
    )


def is_non_placeholder(value: str) -> bool:
    return value.strip().lower() not in PLACEHOLDER_TOKENS


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json_required(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
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
            raise SystemExit(f"{label} is missing a header row: {path}")
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


if __name__ == "__main__":
    main()
