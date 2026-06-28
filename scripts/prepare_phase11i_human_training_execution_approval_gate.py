"""Prepare the Phase 11I human training execution approval gate without executing training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "11I"
STATUS_BLOCKED = "phase11i_blocked_waiting_human_training_execution_approval"
STATUS_PASSED = "phase11i_human_training_execution_approval_passed"
NEXT_BLOCKED = "collect_real_human_training_execution_approval"
NEXT_PASSED = "phase11j_execute_approved_kaggle_training"
REPO_ROOT_REQUIRED_DIRS = ("artifacts", "docs", "scripts", "src")
OUTPUT_DIR_DEFAULT = "artifacts/phase11i_human_training_execution_approval_gate"
PHASE11H_SUMMARY_DEFAULT = (
    "artifacts/phase11h_kaggle_manual_preflight_validation/"
    "phase11h_kaggle_manual_preflight_validation_summary.json"
)
PHASE11H_GATE_DEFAULT = (
    "artifacts/phase11h_kaggle_manual_preflight_validation/phase11h_gate_decision.json"
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
    "example_value",
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
        description="Prepare the Phase 11I human training execution approval gate package."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--phase11h-summary", default=PHASE11H_SUMMARY_DEFAULT)
    parser.add_argument("--phase11h-gate", default=PHASE11H_GATE_DEFAULT)
    parser.add_argument("--approval-csv", default="")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    phase11h_summary_path = (repo_root / args.phase11h_summary).resolve()
    phase11h_gate_path = (repo_root / args.phase11h_gate).resolve()
    approval_csv_path = (repo_root / args.approval_csv).resolve() if args.approval_csv else None
    output_dir = (repo_root / args.output_dir).resolve()

    ensure_repo_root(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase11h_summary = read_json_required(phase11h_summary_path, "Phase 11H summary")
    phase11h_gate = read_json_required(phase11h_gate_path, "Phase 11H gate decision")
    validate_phase11h_inputs(phase11h_summary, phase11h_gate)

    package = build_package(
        repo_root=repo_root,
        output_dir=output_dir,
        phase11h_summary=phase11h_summary,
        phase11h_gate=phase11h_gate,
        phase11h_summary_path=phase11h_summary_path,
        phase11h_gate_path=phase11h_gate_path,
        approval_csv_path=approval_csv_path,
    )
    write_outputs(output_dir, package)
    print(json.dumps(package["gate_decision"], indent=2))


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


def validate_phase11h_inputs(summary: dict[str, Any], gate: dict[str, Any]) -> None:
    failures: list[str] = []
    validate_phase11h_payload(summary, "phase11h_summary", failures)
    validate_phase11h_payload(gate, "phase11h_gate", failures)

    if summary.get("dataset_root") != gate.get("dataset_root"):
        failures.append("Phase 11H summary and gate must agree on dataset_root")
    if summary.get("yaml_path") != gate.get("yaml_path"):
        failures.append("Phase 11H summary and gate must agree on yaml_path")
    if summary.get("status") != gate.get("status"):
        failures.append("Phase 11H summary and gate must agree on status")

    if failures:
        raise SystemExit("Phase 11H validation failed:\n- " + "\n- ".join(failures))


def validate_phase11h_payload(payload: dict[str, Any], label: str, failures: list[str]) -> None:
    expected_pairs = {
        "status": "phase11h_kaggle_manual_preflight_validation_passed",
        "ready_for_training_execution_candidate": True,
        "ready_for_training_execution": False,
        "training_execution_approval_required": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
    }
    for key, expected in expected_pairs.items():
        if payload.get(key) != expected:
            failures.append(f"{label}.{key} must equal {expected!r}; got {payload.get(key)!r}")

    mutation_fields = [
        "original_dataset_mutated",
        "staging_dataset_mutated",
        "kaggle_dataset_uploaded",
        "weights_or_checkpoints_created",
    ]
    for field in mutation_fields:
        if payload.get(field) is not False:
            failures.append(f"{label}.{field} must remain false")


def build_package(
    repo_root: Path,
    output_dir: Path,
    phase11h_summary: dict[str, Any],
    phase11h_gate: dict[str, Any],
    phase11h_summary_path: Path,
    phase11h_gate_path: Path,
    approval_csv_path: Path | None,
) -> dict[str, Any]:
    template_row = build_approval_template_row(
        dataset_root=str(phase11h_summary["dataset_root"]),
        yaml_path=str(phase11h_summary["yaml_path"]),
    )
    approval_template_rows = [template_row]
    checklist_text = build_checklist()
    approval_validation = validate_optional_approval_csv(
        approval_csv_path=approval_csv_path,
        expected_dataset_root=str(phase11h_summary["dataset_root"]),
        expected_yaml_path=str(phase11h_summary["yaml_path"]),
    )

    gate_decision = build_gate_decision(
        repo_root=repo_root,
        output_dir=output_dir,
        phase11h_summary=phase11h_summary,
        phase11h_gate=phase11h_gate,
        phase11h_summary_path=phase11h_summary_path,
        phase11h_gate_path=phase11h_gate_path,
        approval_validation=approval_validation,
    )
    handoff_summary = build_handoff_summary(
        repo_root=repo_root,
        output_dir=output_dir,
        phase11h_summary=phase11h_summary,
        phase11h_summary_path=phase11h_summary_path,
        phase11h_gate_path=phase11h_gate_path,
        approval_validation=approval_validation,
        gate_decision=gate_decision,
    )
    non_execution_manifest = build_non_execution_manifest(approval_validation, gate_decision)
    readme_text = build_readme(gate_decision, approval_validation)

    return {
        "approval_template_rows": approval_template_rows,
        "checklist_text": checklist_text,
        "gate_decision": gate_decision,
        "handoff_summary": handoff_summary,
        "non_execution_manifest": non_execution_manifest,
        "readme_text": readme_text,
    }


def build_approval_template_row(dataset_root: str, yaml_path: str) -> dict[str, str]:
    return {
        "approval_id": "",
        "human_approver": "",
        "approval_datetime": "",
        "approved_for_training_execution": "false",
        "dataset_root": dataset_root,
        "yaml_path": yaml_path,
        "training_command": "",
        "expected_output_dir": "",
        "acknowledge_no_dataset_mutation": "false",
        "acknowledge_training_will_create_weights": "false",
        "notes": "Explicit human approval is required before any later training execution phase.",
    }


def build_checklist() -> str:
    return """# Phase 11I Human Training Execution Approval Checklist

- [ ] Phase 11H summary exists and status is `phase11h_kaggle_manual_preflight_validation_passed`.
- [ ] Phase 11H gate decision exists and confirms Kaggle dataset/YAML structure passed.
- [ ] `ready_for_training_execution_candidate` is `true`.
- [ ] `ready_for_training_execution` is still `false` before human approval.
- [ ] `training_execution_approval_required` is `true`.
- [ ] No training, evaluation, or inference has been executed.
- [ ] No dataset mutation or Kaggle upload has occurred.
- [ ] Approval CSV contains exactly one real human approval row.
- [ ] Approval row explicitly sets `approved_for_training_execution=true`.
- [ ] Approval row matches the Phase 11H `dataset_root` and `yaml_path`.
- [ ] Approval row acknowledges no dataset mutation.
- [ ] Approval row acknowledges that later training will create weights/checkpoints.
- [ ] Phase 11I itself remains gate-only and record-only even if the approval passes.
"""


def validate_optional_approval_csv(
    approval_csv_path: Path | None,
    expected_dataset_root: str,
    expected_yaml_path: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provided": approval_csv_path is not None,
        "path": str(approval_csv_path) if approval_csv_path else "",
        "path_exists": approval_csv_path.exists() if approval_csv_path else False,
        "valid": False,
        "approval_row": None,
        "errors": [],
    }
    if approval_csv_path is None:
        return result
    if not approval_csv_path.exists():
        result["errors"].append(f"Approval CSV not found: {approval_csv_path}")
        return result

    rows = read_csv_rows(approval_csv_path)
    if len(rows) != 1:
        result["errors"].append(f"Approval CSV must contain exactly one row; found {len(rows)}")
        return result

    row = rows[0]
    result["approval_row"] = row
    for field in APPROVAL_FIELDS:
        if field not in row:
            result["errors"].append(f"Approval CSV missing required field: {field}")
    if result["errors"]:
        return result

    require_true_field(row, "approved_for_training_execution", result["errors"])
    require_true_field(row, "acknowledge_no_dataset_mutation", result["errors"])
    require_true_field(row, "acknowledge_training_will_create_weights", result["errors"])
    require_non_empty_field(row, "training_command", result["errors"])
    require_non_empty_field(row, "expected_output_dir", result["errors"])
    require_non_empty_field(row, "human_approver", result["errors"])
    require_non_empty_field(row, "approval_datetime", result["errors"])
    require_non_empty_field(row, "approval_id", result["errors"])

    if row.get("dataset_root", "").strip() != expected_dataset_root:
        result["errors"].append("Approval CSV dataset_root must match Phase 11H dataset_root exactly")
    if row.get("yaml_path", "").strip() != expected_yaml_path:
        result["errors"].append("Approval CSV yaml_path must match Phase 11H yaml_path exactly")

    placeholder_fields = [
        "approval_id",
        "human_approver",
        "approval_datetime",
        "training_command",
        "expected_output_dir",
        "notes",
    ]
    for field in placeholder_fields:
        if is_placeholder_value(row.get(field, "")):
            result["errors"].append(f"Approval CSV field {field} contains placeholder/example content")

    if result["errors"]:
        return result
    result["valid"] = True
    return result


def require_true_field(row: dict[str, str], field: str, errors: list[str]) -> None:
    if not normalize_bool(row.get(field, "")):
        errors.append(f"Approval CSV field {field} must be true")


def require_non_empty_field(row: dict[str, str], field: str, errors: list[str]) -> None:
    if not row.get(field, "").strip():
        errors.append(f"Approval CSV field {field} must be non-empty")


def is_placeholder_value(value: str) -> bool:
    normalized = " ".join(value.strip().lower().split())
    if normalized in PLACEHOLDER_TOKENS:
        return True
    return "example" in normalized or "placeholder" in normalized or normalized.startswith("todo")


def normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_gate_decision(
    repo_root: Path,
    output_dir: Path,
    phase11h_summary: dict[str, Any],
    phase11h_gate: dict[str, Any],
    phase11h_summary_path: Path,
    phase11h_gate_path: Path,
    approval_validation: dict[str, Any],
) -> dict[str, Any]:
    passed = bool(approval_validation["valid"])
    status = STATUS_PASSED if passed else STATUS_BLOCKED
    next_allowed_step = NEXT_PASSED if passed else NEXT_BLOCKED
    return {
        "phase": PHASE,
        "status": status,
        "phase11h_status": phase11h_summary["status"],
        "phase11h_summary_path": normalize_relpath(repo_root, phase11h_summary_path),
        "phase11h_gate_path": normalize_relpath(repo_root, phase11h_gate_path),
        "approval_csv_provided": approval_validation["provided"],
        "approval_csv_path": normalize_optional_path(repo_root, approval_validation["path"]),
        "approval_csv_valid": passed,
        "approval_validation_errors": approval_validation["errors"],
        "training_allowed": passed,
        "ready_for_training_execution_candidate": True,
        "ready_for_training_execution": passed,
        "training_execution_approval_required": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created": False,
        "weights_or_checkpoints_created": False,
        "dataset_root": phase11h_summary["dataset_root"],
        "yaml_path": phase11h_summary["yaml_path"],
        "manual_kaggle_preflight_structural_passed": phase11h_gate["manual_kaggle_preflight_structural_passed"],
        "next_allowed_step": next_allowed_step,
        "outputs": {
            "approval_template_csv": str(output_dir / "phase11i_training_execution_approval_template.csv"),
            "checklist_md": str(output_dir / "phase11i_training_execution_checklist.md"),
            "gate_decision_json": str(output_dir / "phase11i_gate_decision.json"),
            "handoff_summary_json": str(output_dir / "phase11i_handoff_summary.json"),
            "non_execution_manifest_json": str(output_dir / "phase11i_non_execution_manifest.json"),
            "readme_md": str(output_dir / "README.md"),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_handoff_summary(
    repo_root: Path,
    output_dir: Path,
    phase11h_summary: dict[str, Any],
    phase11h_summary_path: Path,
    phase11h_gate_path: Path,
    approval_validation: dict[str, Any],
    gate_decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": gate_decision["status"],
        "mode": "gate_only_record_only_human_training_execution_approval",
        "source_phase11h_summary": normalize_relpath(repo_root, phase11h_summary_path),
        "source_phase11h_gate": normalize_relpath(repo_root, phase11h_gate_path),
        "approval_csv_provided": approval_validation["provided"],
        "approval_csv_valid": approval_validation["valid"],
        "dataset_root": phase11h_summary["dataset_root"],
        "yaml_path": phase11h_summary["yaml_path"],
        "ready_for_training_execution_candidate": True,
        "ready_for_training_execution": gate_decision["ready_for_training_execution"],
        "training_allowed": gate_decision["training_allowed"],
        "training_execution_approval_required": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created": False,
        "next_allowed_step": gate_decision["next_allowed_step"],
        "output_dir": str(output_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_non_execution_manifest(
    approval_validation: dict[str, Any],
    gate_decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": gate_decision["status"],
        "approval_csv_valid": approval_validation["valid"],
        "no_training": True,
        "no_evaluation": True,
        "no_inference": True,
        "no_dataset_mutation": True,
        "no_kaggle_upload": True,
        "no_weights_or_checkpoints": True,
        "notes": "Phase 11I is a human approval gate only. It records approval state without executing training.",
    }


def build_readme(gate_decision: dict[str, Any], approval_validation: dict[str, Any]) -> str:
    approval_section = [
        f"- approval_csv_provided = `{gate_decision['approval_csv_provided']}`",
        f"- approval_csv_valid = `{gate_decision['approval_csv_valid']}`",
    ]
    if approval_validation["errors"]:
        approval_section.append(f"- approval_validation_errors = `{len(approval_validation['errors'])}`")
    return "\n".join(
        [
            "# Phase 11I Human Training Execution Approval Gate",
            "",
            "Phase 11I is a strict gate-only and record-only human approval phase after Phase 11H Kaggle manual preflight validation.",
            "",
            "## Outcome",
            f"- status = `{gate_decision['status']}`",
            f"- training_allowed = `{gate_decision['training_allowed']}`",
            f"- ready_for_training_execution = `{gate_decision['ready_for_training_execution']}`",
            f"- ready_for_training_execution_candidate = `{gate_decision['ready_for_training_execution_candidate']}`",
            f"- training_execution_approval_required = `{gate_decision['training_execution_approval_required']}`",
            f"- training_executed = `{gate_decision['training_executed']}`",
            f"- evaluation_executed = `{gate_decision['evaluation_executed']}`",
            f"- inference_executed = `{gate_decision['inference_executed']}`",
            f"- next_allowed_step = `{gate_decision['next_allowed_step']}`",
            "",
            "## Approval State",
            *approval_section,
            "",
            "## Guardrails",
            "- No training is executed in Phase 11I.",
            "- No evaluation or inference is executed in Phase 11I.",
            "- No dataset mutation or Kaggle upload is performed in Phase 11I.",
            "- No weights or checkpoints are created in Phase 11I.",
        ]
    ) + "\n"


def write_outputs(output_dir: Path, package: dict[str, Any]) -> None:
    write_csv(
        output_dir / "phase11i_training_execution_approval_template.csv",
        package["approval_template_rows"],
        APPROVAL_FIELDS,
    )
    (output_dir / "phase11i_training_execution_checklist.md").write_text(
        package["checklist_text"], encoding="utf-8"
    )
    write_json(output_dir / "phase11i_gate_decision.json", package["gate_decision"])
    write_json(output_dir / "phase11i_handoff_summary.json", package["handoff_summary"])
    write_json(output_dir / "phase11i_non_execution_manifest.json", package["non_execution_manifest"])
    (output_dir / "README.md").write_text(package["readme_text"], encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def normalize_optional_path(repo_root: Path, path_value: str) -> str:
    if not path_value:
        return ""
    return normalize_relpath(repo_root, Path(path_value))


if __name__ == "__main__":
    main()
