"""Prepare the Phase 11J.0 approved Kaggle training command lock without executing training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "11J.0"
STATUS = "phase11j0_approved_kaggle_training_command_locked"
NEXT_ALLOWED_STEP = "phase11j1_execute_locked_kaggle_training"
REPO_ROOT_REQUIRED_DIRS = ("artifacts", "docs", "scripts", "src")
OUTPUT_DIR_DEFAULT = "artifacts/phase11j_approved_kaggle_training_command_lock"
PHASE11I_SUMMARY_DEFAULT = (
    "artifacts/phase11i_human_training_execution_approval_gate/phase11i_handoff_summary.json"
)
PHASE11I_GATE_DEFAULT = (
    "artifacts/phase11i_human_training_execution_approval_gate/phase11i_gate_decision.json"
)
APPROVAL_CSV_DEFAULT = (
    "artifacts/phase11i_human_training_execution_approval_gate/phase11i_training_execution_approval_filled.csv"
)
APPROVAL_FIELDS = [
    "approval_id",
    "human_approver",
    "approval_datetime",
    "approved_for_training_execution",
    "dataset_root",
    "yaml_path",
    "training_command",
    "expected_output_dir",
    "acknowledge_no_dataset_mutation",
    "acknowledge_training_will_create_weights",
    "notes",
]
PLACEHOLDER_TOKENS = {
    "",
    "placeholder",
    "example",
    "example command",
    "example path",
    "fill manually",
    "manual_reviewer",
    "human_approver",
    "pending",
    "tbd",
    "todo",
    "unknown",
    "n/a",
    "na",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Phase 11J.0 approved Kaggle training command lock."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--phase11i-summary", default=PHASE11I_SUMMARY_DEFAULT)
    parser.add_argument("--phase11i-gate", default=PHASE11I_GATE_DEFAULT)
    parser.add_argument("--approval-csv", default=APPROVAL_CSV_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    phase11i_summary_path = (repo_root / args.phase11i_summary).resolve()
    phase11i_gate_path = (repo_root / args.phase11i_gate).resolve()
    approval_csv_path = (repo_root / args.approval_csv).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    ensure_repo_root(repo_root)
    phase11i_summary = read_json_required(phase11i_summary_path, "Phase 11I handoff summary")
    phase11i_gate = read_json_required(phase11i_gate_path, "Phase 11I gate decision")
    validate_phase11i_inputs(phase11i_summary, phase11i_gate)

    approval_row = validate_approval_csv(
        approval_csv_path=approval_csv_path,
        expected_dataset_root=str(phase11i_gate["dataset_root"]),
        expected_yaml_path=str(phase11i_gate["yaml_path"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    package = build_package(
        repo_root=repo_root,
        output_dir=output_dir,
        phase11i_summary=phase11i_summary,
        phase11i_gate=phase11i_gate,
        phase11i_summary_path=phase11i_summary_path,
        phase11i_gate_path=phase11i_gate_path,
        approval_csv_path=approval_csv_path,
        approval_row=approval_row,
    )
    write_outputs(output_dir, package)
    print(json.dumps(package["summary"], indent=2))


def ensure_repo_root(repo_root: Path) -> None:
    missing = [name for name in REPO_ROOT_REQUIRED_DIRS if not (repo_root / name).exists()]
    if missing:
        raise SystemExit(f"Repo root validation failed; missing required paths: {', '.join(missing)}")


def read_json_required(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path} ({exc})") from exc


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_phase11i_inputs(summary: dict[str, Any], gate: dict[str, Any]) -> None:
    failures: list[str] = []
    validate_phase11i_payload(summary, "phase11i_summary", failures)
    validate_phase11i_payload(gate, "phase11i_gate", failures)

    if summary.get("dataset_root") != gate.get("dataset_root"):
        failures.append("Phase 11I summary and gate must agree on dataset_root")
    if summary.get("yaml_path") != gate.get("yaml_path"):
        failures.append("Phase 11I summary and gate must agree on yaml_path")
    if summary.get("status") != gate.get("status"):
        failures.append("Phase 11I summary and gate must agree on status")

    if failures:
        raise SystemExit("Phase 11I validation failed:\n- " + "\n- ".join(failures))


def validate_phase11i_payload(payload: dict[str, Any], label: str, failures: list[str]) -> None:
    expected = {
        "status": "phase11i_human_training_execution_approval_passed",
        "approval_csv_valid": True,
        "training_allowed": True,
        "ready_for_training_execution": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "kaggle_upload_executed": False,
        "next_allowed_step": "phase11j_execute_approved_kaggle_training",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            failures.append(f"{label}.{key} must equal {value!r}; got {payload.get(key)!r}")

    if payload.get("weights_created") is not False and payload.get("weights_or_checkpoints_created") is not False:
        failures.append(f"{label} must keep weights_created or weights_or_checkpoints_created false")


def validate_approval_csv(
    approval_csv_path: Path,
    expected_dataset_root: str,
    expected_yaml_path: str,
) -> dict[str, str]:
    if not approval_csv_path.exists():
        raise SystemExit(f"Approval CSV not found: {approval_csv_path}")
    rows = read_csv_rows(approval_csv_path)
    failures: list[str] = []

    if len(rows) != 1:
        failures.append(f"Approval CSV must contain exactly one row; found {len(rows)}")
    row = rows[0] if rows else {}

    for field in APPROVAL_FIELDS:
        if field not in row:
            failures.append(f"Approval CSV missing required field: {field}")
    if failures:
        raise SystemExit("Approval CSV validation failed:\n- " + "\n- ".join(failures))

    require_true_field(row, "approved_for_training_execution", failures)
    require_true_field(row, "acknowledge_no_dataset_mutation", failures)
    require_true_field(row, "acknowledge_training_will_create_weights", failures)
    require_non_empty_field(row, "approval_id", failures)
    require_non_empty_field(row, "human_approver", failures)
    require_non_empty_field(row, "approval_datetime", failures)
    require_non_empty_field(row, "training_command", failures)
    require_non_empty_field(row, "expected_output_dir", failures)

    if row.get("dataset_root", "").strip() != expected_dataset_root:
        failures.append("Approval CSV dataset_root must match Phase 11I dataset_root exactly")
    if row.get("yaml_path", "").strip() != expected_yaml_path:
        failures.append("Approval CSV yaml_path must match Phase 11I yaml_path exactly")

    for field in ("approval_id", "human_approver", "training_command", "expected_output_dir"):
        if is_placeholder_value(row.get(field, "")):
            failures.append(f"Approval CSV field {field} contains placeholder/example content")
    if not looks_like_timestamp(row.get("approval_datetime", "")):
        failures.append("Approval CSV approval_datetime must be a parseable timestamp-like string")

    if failures:
        raise SystemExit("Approval CSV validation failed:\n- " + "\n- ".join(failures))
    return row


def require_true_field(row: dict[str, str], field: str, failures: list[str]) -> None:
    if not normalize_bool(row.get(field, "")):
        failures.append(f"Approval CSV field {field} must be true")


def require_non_empty_field(row: dict[str, str], field: str, failures: list[str]) -> None:
    if not row.get(field, "").strip():
        failures.append(f"Approval CSV field {field} must be non-empty")


def normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def is_placeholder_value(value: str) -> bool:
    normalized = " ".join(value.strip().lower().split())
    if normalized in PLACEHOLDER_TOKENS:
        return True
    return "example" in normalized or "placeholder" in normalized or normalized.startswith("todo")


def looks_like_timestamp(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    normalized = text.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
        return True
    except ValueError:
        pass
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return True
    return False


def build_package(
    repo_root: Path,
    output_dir: Path,
    phase11i_summary: dict[str, Any],
    phase11i_gate: dict[str, Any],
    phase11i_summary_path: Path,
    phase11i_gate_path: Path,
    approval_csv_path: Path,
    approval_row: dict[str, str],
) -> dict[str, Any]:
    approved_command = approval_row["training_command"]
    expected_output_dir = approval_row["expected_output_dir"]
    summary = build_summary(
        repo_root=repo_root,
        phase11i_summary=phase11i_summary,
        phase11i_gate=phase11i_gate,
        phase11i_summary_path=phase11i_summary_path,
        phase11i_gate_path=phase11i_gate_path,
        approval_csv_path=approval_csv_path,
        approval_row=approval_row,
    )
    expected_outputs = build_expected_outputs(expected_output_dir)
    integrity_rows = build_integrity_rows(approval_csv_path, approval_row, approved_command, expected_output_dir)
    non_execution_manifest = build_non_execution_manifest()
    readme_text = build_readme(summary, approval_csv_path)
    return {
        "summary": summary,
        "approved_training_command": approved_command,
        "expected_outputs": expected_outputs,
        "integrity_rows": integrity_rows,
        "non_execution_manifest": non_execution_manifest,
        "readme_text": readme_text,
    }


def build_summary(
    repo_root: Path,
    phase11i_summary: dict[str, Any],
    phase11i_gate: dict[str, Any],
    phase11i_summary_path: Path,
    phase11i_gate_path: Path,
    approval_csv_path: Path,
    approval_row: dict[str, str],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": STATUS,
        "phase11i_status": "phase11i_human_training_execution_approval_passed",
        "phase11i_summary_path": normalize_relpath(repo_root, phase11i_summary_path),
        "phase11i_gate_path": normalize_relpath(repo_root, phase11i_gate_path),
        "approval_csv_path": normalize_relpath(repo_root, approval_csv_path),
        "approval_csv_valid": True,
        "training_command_locked": True,
        "approved_training_command": approval_row["training_command"],
        "expected_output_dir": approval_row["expected_output_dir"],
        "dataset_root": approval_row["dataset_root"],
        "yaml_path": approval_row["yaml_path"],
        "training_allowed": True,
        "ready_for_training_execution": True,
        "ready_for_training_execution_candidate": bool(phase11i_summary.get("ready_for_training_execution_candidate")),
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created": False,
        "weights_or_checkpoints_created": False,
        "next_allowed_step": NEXT_ALLOWED_STEP,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_expected_outputs(expected_output_dir: str) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "expected_output_dir": expected_output_dir,
        "expected_runtime_outputs": [
            {"path": expected_output_dir, "kind": "kaggle_working_output_directory"},
            {"path": f"{expected_output_dir}/results.csv", "kind": "training_results_csv"},
            {"path": f"{expected_output_dir}/weights/best.pt", "kind": "best_weights_checkpoint"},
            {"path": f"{expected_output_dir}/weights/last.pt", "kind": "last_weights_checkpoint"},
            {"path": f"{expected_output_dir}/args.yaml", "kind": "yolo_args_or_config_if_produced"},
            {"path": f"{expected_output_dir}/*.log", "kind": "training_logs_if_produced"},
            {"path": f"{expected_output_dir}/events*", "kind": "run_metadata_files_if_produced"},
        ],
        "notes": "These are expected outputs for Phase 11J.1 runtime execution only. Phase 11J.0 does not create them.",
    }


def build_integrity_rows(
    approval_csv_path: Path,
    approval_row: dict[str, str],
    approved_command: str,
    expected_output_dir: str,
) -> list[dict[str, str]]:
    return [
        integrity_row(
            "approval_csv_exists",
            "true",
            "existing approval csv",
            str(approval_csv_path),
            "Phase 11J.0 requires a concrete approval CSV provenance source.",
        ),
        integrity_row(
            "single_approval_row",
            "true",
            "exactly one row",
            "1",
            "Phase 11J.0 locks one explicit approved command only.",
        ),
        integrity_row(
            "approved_for_training_execution",
            "true",
            "true",
            approval_row["approved_for_training_execution"],
            "The approval row must explicitly authorize training execution.",
        ),
        integrity_row(
            "training_command_locked_exactly",
            "true",
            "exact command from approval csv",
            approved_command,
            "Phase 11J.0 stores the approved command exactly as provided.",
        ),
        integrity_row(
            "expected_output_dir_locked_exactly",
            "true",
            "exact expected output dir from approval csv",
            expected_output_dir,
            "Phase 11J.0 stores the expected output directory exactly as approved.",
        ),
    ]


def build_non_execution_manifest() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": STATUS,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created": False,
        "checkpoints_created": False,
        "notes": "This phase only locks the approved command and expected outputs. It does not execute training.",
    }


def build_readme(summary: dict[str, Any], approval_csv_path: Path) -> str:
    return "\n".join(
        [
            "# Phase 11J.0 Approved Kaggle Training Command Lock",
            "",
            "Phase 11J.0 locks the exact human-approved Kaggle training command without executing it.",
            "",
            "## Outcome",
            f"- status = `{summary['status']}`",
            f"- approval_csv_valid = `{summary['approval_csv_valid']}`",
            f"- training_command_locked = `{summary['training_command_locked']}`",
            f"- training_allowed = `{summary['training_allowed']}`",
            f"- ready_for_training_execution = `{summary['ready_for_training_execution']}`",
            f"- training_executed = `{summary['training_executed']}`",
            f"- evaluation_executed = `{summary['evaluation_executed']}`",
            f"- inference_executed = `{summary['inference_executed']}`",
            f"- next_allowed_step = `{summary['next_allowed_step']}`",
            "",
            "## Provenance",
            f"- approval_csv_path = `{approval_csv_path}`",
            f"- expected_output_dir = `{summary['expected_output_dir']}`",
            "",
            "## Guardrails",
            "- No training is executed in Phase 11J.0.",
            "- No evaluation or inference is executed in Phase 11J.0.",
            "- No dataset mutation or Kaggle upload is performed in Phase 11J.0.",
            "- No weights or checkpoints are created in Phase 11J.0.",
        ]
    ) + "\n"


def integrity_row(check_id: str, passed: str, expected: str, observed: str, notes: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "passed": passed,
        "expected": expected,
        "observed": observed,
        "notes": notes,
    }


def write_outputs(output_dir: Path, package: dict[str, Any]) -> None:
    write_json(output_dir / "phase11j0_training_command_lock_summary.json", package["summary"])
    (output_dir / "phase11j0_approved_training_command.txt").write_text(
        package["approved_training_command"] + "\n", encoding="utf-8"
    )
    write_json(output_dir / "phase11j0_expected_outputs.json", package["expected_outputs"])
    write_csv(output_dir / "phase11j0_command_integrity_check.csv", package["integrity_rows"])
    write_json(output_dir / "phase11j0_non_execution_manifest.json", package["non_execution_manifest"])
    (output_dir / "README.md").write_text(package["readme_text"], encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
