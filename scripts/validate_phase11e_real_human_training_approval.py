"""Validate real human training approval contents without executing training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "11E-real"
PASS_STATUS = "phase11e_real_human_training_approval_validated"
FAIL_STATUS = "phase11e_real_human_training_approval_rejected_or_incomplete"
PASS_NEXT_STEP = "phase11f_prepare_approved_staging_training_execution_package_no_training_yet"
FAIL_NEXT_STEP = "fix_real_approval_contents_or_finalize_audit_only_paper"
REPO_ROOT_REQUIRED_DIRS = ("artifacts", "docs", "scripts", "src")
OUTPUT_DIR_DEFAULT = "artifacts/phase11e_real_approval_validation"
DEFAULT_APPROVAL_FILE = "artifacts/phase11d_approval_or_no_training_gate/phase11d_real_approval_template.csv"
EXPECTED_CONFIG_REL = "configs/train/phase9v_staging_chipped_yolov8n.yaml"
EXPECTED_SCRIPT_REL = "scripts/execute_phase9y_approved_staging_training.py"
EXPECTED_STAGING_COPY_REL = "artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_copy"
EXPECTED_STAGING_YAML_REL = "artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml"
PLACEHOLDER_TOKENS = {
    "",
    "fill manually",
    "manual_reviewer",
    "placeholder",
    "pending",
    "unknown",
    "n/a",
    "na",
    "tbd",
    "todo",
    "example",
    "example fixture",
    "fixture",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a real human training approval CSV by content.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--approval-file", default=DEFAULT_APPROVAL_FILE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    approval_path = (repo_root / args.approval_file).resolve()

    ensure_repo_root(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_validation(repo_root=repo_root, approval_path=approval_path)
    write_outputs(output_dir=output_dir, results=results)
    print(json.dumps(results["summary"], indent=2))


def evaluate_validation(repo_root: Path, approval_path: Path) -> dict[str, Any]:
    phase11d_template_path = repo_root / DEFAULT_APPROVAL_FILE

    phase11d_summary = read_json_required(
        repo_root / "artifacts/phase11d_approval_or_no_training_gate/phase11d_gate_summary.json",
        "phase11d_gate_summary.json",
    )
    template_rows = read_csv_rows(phase11d_template_path)
    training_allowed_conditions = read_text_required(
        repo_root / "artifacts/phase11d_approval_or_no_training_gate/phase11d_training_allowed_conditions.md"
    )
    approval_evidence_requirements = read_text_required(
        repo_root / "artifacts/phase11d_approval_or_no_training_gate/phase11d_approval_evidence_requirements.md"
    )
    phase11c_after_approval = read_text_required(
        repo_root / "artifacts/phase11c_paper_direction_experiment_plan/phase11c_experiment_plan_after_approval.md"
    )
    phase9z5_summary = read_optional_json(
        repo_root / "artifacts/phase9z5_real_human_approval_rerun_chipped/phase9z5_real_human_approval_rerun_summary.json"
    )
    phase9v_summary = read_optional_json(
        repo_root / "artifacts/phase9v_staging_training_config_chipped/phase9v_training_config_summary.json"
    )
    phase9w_summary = read_optional_json(
        repo_root / "artifacts/phase9w_final_command_review_chipped/phase9w_final_command_review_summary.json"
    )
    phase9y_summary = read_optional_json(
        repo_root / "artifacts/phase9y_approved_staging_training_chipped/phase9y_training_execution_summary.json"
    )
    phase9y_command = read_optional_text(
        repo_root / "artifacts/phase9y_approved_staging_training_chipped/phase9y_training_command.txt"
    )
    experiment_log = read_text_required(repo_root / "docs/experiment_log.md")

    phase11d_validated = validate_phase11d_summary(phase11d_summary)
    report_rows: list[dict[str, str]] = []
    rejection_rows: list[dict[str, str]] = []

    candidate_exists = approval_path.exists()
    report_rows.append(
        check_row(
            "approval file exists",
            candidate_exists,
            "existing approval CSV",
            approval_path.relative_to(repo_root).as_posix() if candidate_exists else "missing",
            "Phase 11E-real requires a concrete approval file path to inspect by content.",
        )
    )

    approval_row: dict[str, str] = {}
    if candidate_exists:
        try:
            approval_rows = read_csv_rows(approval_path)
        except Exception as exc:  # pragma: no cover - surfaced as report
            approval_rows = []
            rejection_rows.append(
                rejection_reason("parse_failure", f"Failed to read approval CSV: {exc}", approval_path.relative_to(repo_root).as_posix())
            )
        if len(approval_rows) != 1:
            rejection_rows.append(
                rejection_reason(
                    "row_count_invalid",
                    f"Expected exactly one approval row, found {len(approval_rows)}.",
                    approval_path.relative_to(repo_root).as_posix(),
                )
            )
            approval_row = approval_rows[0] if approval_rows else {}
        else:
            approval_row = approval_rows[0]
    else:
        rejection_rows.append(rejection_reason("missing_file", "Approval file is missing.", args.approval_file))

    expected_config = normalize_path_string(
        phase9v_summary.get("training_config_path") if phase9v_summary else str(repo_root / EXPECTED_CONFIG_REL),
        repo_root,
    )
    expected_script = normalize_path_string(str(repo_root / EXPECTED_SCRIPT_REL), repo_root)
    expected_dataset = normalize_path_string(str(repo_root / EXPECTED_STAGING_COPY_REL), repo_root)
    expected_yaml = normalize_path_string(str(repo_root / EXPECTED_STAGING_YAML_REL), repo_root)
    expected_command = normalize_command_string(phase9y_command or "")

    criteria_rows = build_acceptance_criteria(
        approval_row=approval_row,
        template_row_defaults=default_phase11d_template_row(),
        expected_config=expected_config,
        expected_script=expected_script,
        expected_dataset=expected_dataset,
        expected_yaml=expected_yaml,
        expected_command=expected_command,
        report_rows=report_rows,
        rejection_rows=rejection_rows,
    )

    all_passed = candidate_exists and all(row["passed"] == "true" for row in criteria_rows)
    status = PASS_STATUS if all_passed else FAIL_STATUS
    next_allowed_step = PASS_NEXT_STEP if all_passed else FAIL_NEXT_STEP
    historical_phase9z5_training_allowed = phase9z5_summary.get("training_allowed") if phase9z5_summary else None
    historical_phase9z5_approval_validated = phase9z5_summary.get("approval_validated") if phase9z5_summary else None
    historical_prior_gate_blocked = bool(
        phase9z5_summary
        and (
            phase9z5_summary.get("training_allowed") is False
            or phase9z5_summary.get("approval_validated") is False
        )
    )

    approved_for_staging_training = normalize_yes_no(approval_row.get("approved_for_staging_training", "")) if approval_row else False
    approved_dataset = approval_row.get("approved_dataset_or_staging_copy", "") if approval_row else ""
    approved_config = approval_row.get("approved_config_path", "") if approval_row else ""
    approved_command = approval_row.get("approved_command_or_script", "") if approval_row else ""

    validated_training_scope_md = build_validated_training_scope_markdown(
        all_passed=all_passed,
        approval_path=approval_path,
        approved_for_staging_training=approved_for_staging_training,
        approved_dataset=approved_dataset,
        approved_config=approved_config,
        approved_command=approved_command,
        expected_dataset=expected_dataset,
        expected_yaml=expected_yaml,
        expected_config=expected_config,
        expected_script=expected_script,
        expected_command=expected_command,
        rejection_rows=rejection_rows,
    )

    non_mutation_manifest = {
        "phase": PHASE,
        "status": status,
        "approval_validation_only": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "approval_state_mutated": False,
        "phase9z5_state_preserved": True,
        "phase9z5_state_mutated": False,
        "notes": "Phase 11E-real validates approval contents only and does not execute or authorize training by itself.",
    }

    summary = {
        "phase": PHASE,
        "status": status,
        "approval_validation_only": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "approval_state_mutated": False,
        "phase11d_validated": phase11d_validated,
        "real_human_approval_file_checked": approval_path.relative_to(repo_root).as_posix(),
        "real_human_approval_validated": all_passed,
        "approved_for_staging_training": approved_for_staging_training,
        "approved_dataset_or_staging_copy": approved_dataset,
        "approved_config_path": approved_config,
        "approved_command_or_script": approved_command,
        "training_allowed_after_phase11e": all_passed,
        "original_dataset_mutation_allowed": False,
        "phase9z5_training_allowed": historical_phase9z5_training_allowed,
        "phase9z5_approval_validated": historical_phase9z5_approval_validated,
        "historical_phase9z5_training_allowed": historical_phase9z5_training_allowed,
        "historical_phase9z5_approval_validated": historical_phase9z5_approval_validated,
        "historical_prior_gate_blocked": historical_prior_gate_blocked,
        "phase9z5_state_mutated": False,
        "rejection_reason_count": len(rejection_rows),
        "template_file_consulted": DEFAULT_APPROVAL_FILE,
        "training_allowed_conditions_consulted": "Training remains disallowed after Phase 11D." in training_allowed_conditions,
        "approval_evidence_requirements_consulted": "A candidate approval file is not valid merely because it exists." in approval_evidence_requirements,
        "phase11c_after_approval_plan_consulted": "conditional" in phase11c_after_approval.lower(),
        "phase9v_summary_consulted": phase9v_summary is not None,
        "phase9w_summary_consulted": phase9w_summary is not None,
        "phase9y_summary_consulted": phase9y_summary is not None,
        "experiment_log_consulted": "Phase 11D" in experiment_log,
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "summary": summary,
        "report_rows": report_rows,
        "criteria_rows": criteria_rows,
        "rejection_rows": rejection_rows,
        "validated_training_scope_md": validated_training_scope_md,
        "non_mutation_manifest": non_mutation_manifest,
    }


def ensure_repo_root(repo_root: Path) -> None:
    missing = [name for name in REPO_ROOT_REQUIRED_DIRS if not (repo_root / name).exists()]
    if missing:
        raise SystemExit(f"Repo root validation failed; missing required paths: {', '.join(missing)}")


def validate_phase11d_summary(summary: dict[str, Any]) -> bool:
    checks = [
        summary.get("status") == "phase11d_approval_or_no_training_gate_prepared",
        summary.get("phase11c_validated") is True,
        summary.get("phase9z5_training_allowed") is False,
        summary.get("phase9z5_approval_validated") is False,
        summary.get("real_approval_template_created") is True,
        summary.get("training_allowed_after_phase11d") is False,
        summary.get("next_allowed_step") == "collect_real_human_training_approval_or_finalize_audit_only_paper",
    ]
    return all(checks)


def build_acceptance_criteria(
    *,
    approval_row: dict[str, str],
    template_row_defaults: dict[str, str],
    expected_config: str,
    expected_script: str,
    expected_dataset: str,
    expected_yaml: str,
    expected_command: str,
    report_rows: list[dict[str, str]],
    rejection_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def require(check_name: str, passed: bool, expected: str, observed: str, notes: str, reason_id: str | None = None, reason: str | None = None) -> None:
        rows.append(check_row(check_name, passed, expected, observed, notes))
        if not passed and reason_id and reason:
            rejection_rows.append(rejection_reason(reason_id, reason, observed))

    fields = [
        "approver_name",
        "approver_role",
        "approval_date",
        "approval_scope",
        "approved_for_staging_training",
        "approved_dataset_or_staging_copy",
        "approved_config_path",
        "approved_command_or_script",
        "approved_expected_outputs",
        "approval_limitations",
        "explicit_human_confirmation",
        "signature_or_confirmation_text",
    ]

    for field in fields:
        observed = approval_row.get(field, "")
        require(
            f"{field} present",
            not is_blank(observed),
            f"non-empty {field}",
            observed or "blank",
            f"{field} must be filled in the candidate approval file.",
            reason_id=f"missing_{field}",
            reason=f"Required approval field {field} is blank.",
        )

    for field in ("approver_name", "approver_role", "approval_scope", "explicit_human_confirmation", "signature_or_confirmation_text"):
        observed = approval_row.get(field, "")
        require(
            f"{field} not placeholder",
            not looks_like_placeholder(observed),
            f"{field} must be real human content",
            observed or "blank",
            f"{field} must not contain placeholder or example-style content.",
            reason_id=f"placeholder_{field}",
            reason=f"{field} contains placeholder/example content.",
        )

    approved_for_training_value = approval_row.get("approved_for_staging_training", "")
    require(
        "approved_for_staging_training affirmative",
        normalize_yes_no(approved_for_training_value),
        "true/yes/approved",
        approved_for_training_value or "blank",
        "Approval must clearly authorize staging training.",
        reason_id="approved_for_staging_training_not_affirmative",
        reason="approved_for_staging_training is not clearly affirmative.",
    )

    explicit_confirmation_value = approval_row.get("explicit_human_confirmation", "")
    require(
        "explicit_human_confirmation affirmative",
        normalize_yes_no(explicit_confirmation_value) or contains_affirmation_phrase(explicit_confirmation_value),
        "explicit human yes/approved confirmation",
        explicit_confirmation_value or "blank",
        "explicit_human_confirmation must clearly confirm real human approval.",
        reason_id="explicit_confirmation_missing_or_ambiguous",
        reason="explicit_human_confirmation is missing or ambiguous.",
    )

    signature_value = approval_row.get("signature_or_confirmation_text", "")
    require(
        "signature_or_confirmation_text meaningful",
        not is_blank(signature_value) and not looks_like_placeholder(signature_value),
        "meaningful signature or confirmation text",
        signature_value or "blank",
        "A real signature or human confirmation text is required.",
        reason_id="signature_missing",
        reason="signature_or_confirmation_text is missing or placeholder.",
    )

    scope_value = approval_row.get("approval_scope", "")
    scope_lower = normalize_text(scope_value)
    require(
        "approval scope matches staging training",
        "staging" in scope_lower and "train" in scope_lower,
        "scope explicitly references staging training",
        scope_value or "blank",
        "Approval scope must match staging-training authorization.",
        reason_id="scope_mismatch",
        reason="approval_scope does not clearly match staging training.",
    )
    require(
        "approval scope forbids original dataset mutation",
        scope_blocks_original_mutation(scope_value),
        "scope either omits original dataset mutation or explicitly forbids it",
        scope_value or "blank",
        "Approval must not authorize mutation of original dataset labels; explicit prohibitions are allowed.",
        reason_id="scope_authorizes_original_mutation",
        reason="approval_scope appears to authorize original dataset mutation.",
    )

    dataset_value = approval_row.get("approved_dataset_or_staging_copy", "")
    dataset_norm = normalize_path_string(dataset_value)
    require(
        "approved dataset targets staging copy or staging yaml",
        dataset_norm.endswith(expected_dataset) or dataset_norm.endswith(expected_yaml) or expected_dataset in dataset_norm or expected_yaml in dataset_norm,
        f"{expected_dataset} or {expected_yaml}",
        dataset_value or "blank",
        "Approval must identify the staging dataset copy or its staging YAML explicitly.",
        reason_id="dataset_scope_mismatch",
        reason="approved_dataset_or_staging_copy does not match the approved staging scope.",
    )
    require(
        "approved dataset does not target original labels",
        "data/yolo_format" not in dataset_norm and "original" not in normalize_text(dataset_value),
        "no original dataset label mutation target",
        dataset_value or "blank",
        "Approval must not target original dataset labels.",
        reason_id="dataset_targets_original",
        reason="approved_dataset_or_staging_copy appears to target the original dataset.",
    )

    config_value = approval_row.get("approved_config_path", "")
    config_norm = normalize_path_string(config_value)
    require(
        "approved config path matches expected config",
        config_norm.endswith(expected_config) or expected_config in config_norm,
        expected_config,
        config_value or "blank",
        "Approval must explicitly name the Phase 9V staging config path.",
        reason_id="config_path_mismatch",
        reason="approved_config_path does not match the expected staging config.",
    )

    command_value = approval_row.get("approved_command_or_script", "")
    command_norm = normalize_path_string(command_value)
    command_text_norm = normalize_command_string(command_value)
    command_matches = (
        command_norm.endswith(expected_script)
        or expected_script in command_norm
        or (".venv/bin/yolo detect train" in command_text_norm and "staging_dataset_drill_bit_yolo.yaml" in command_text_norm)
    )
    require(
        "approved command or script matches expected staging execution target",
        command_matches,
        f"{expected_script} or explicit yolo detect train command for the staging YAML",
        command_value or "blank",
        "Approval must clearly authorize the intended staging-training command or execution script.",
        reason_id="command_mismatch",
        reason="approved_command_or_script does not clearly match the approved staging-training target.",
    )

    expected_outputs_value = approval_row.get("approved_expected_outputs", "")
    require(
        "approved_expected_outputs meaningful",
        not is_blank(expected_outputs_value) and not looks_like_placeholder(expected_outputs_value),
        "non-placeholder expected outputs description",
        expected_outputs_value or "blank",
        "Approval must state expected outputs for the approved run.",
        reason_id="expected_outputs_missing",
        reason="approved_expected_outputs is blank or placeholder.",
    )

    limitations_value = approval_row.get("approval_limitations", "")
    require(
        "approval_limitations meaningful",
        not is_blank(limitations_value) and not looks_like_placeholder(limitations_value),
        "non-placeholder limitation notes",
        limitations_value or "blank",
        "Approval must include limitation notes so the scope is not ambiguous.",
        reason_id="limitations_missing",
        reason="approval_limitations is blank or placeholder.",
    )

    name_value = approval_row.get("approver_name", "")
    role_value = approval_row.get("approver_role", "")
    require(
        "approver identity not example fixture",
        not any(token in normalize_text(name_value + " " + role_value) for token in ("example", "fixture", "test", "manual_reviewer")),
        "real human approver identity",
        f"name={name_value or 'blank'}; role={role_value or 'blank'}",
        "Approver identity must not match example or fixture-style provenance.",
        reason_id="approver_identity_invalid",
        reason="approver_name or approver_role appears to be example/fixture content.",
    )

    if approval_row and template_row_defaults:
        unchanged = all(
            normalize_text(approval_row.get(key, "")) == normalize_text(template_row_defaults.get(key, ""))
            for key in template_row_defaults
        )
        require(
            "approval row differs from template",
            not unchanged,
            "content changed from template defaults",
            "unchanged_template" if unchanged else "changed",
            "A template row without substantive edits is not valid approval.",
            reason_id="template_unchanged",
            reason="Approval row is unchanged from the Phase 11D template.",
        )

    pending_words = ("pending", "fill manually", "must fill", "tbd", "todo")
    combined_text = " ".join(str(v) for v in approval_row.values())
    require(
        "approval content not pending or instructional",
        not any(word in normalize_text(combined_text) for word in pending_words),
        "no pending/template instructional text",
        combined_text[:200] if combined_text else "blank",
        "Approval contents must not remain in pending or template-instruction mode.",
        reason_id="pending_content",
        reason="Approval contents still contain pending or instructional placeholder text.",
    )

    return rows


def default_phase11d_template_row() -> dict[str, str]:
    return {
        "approver_name": "",
        "approver_role": "",
        "approval_date": "",
        "approval_scope": "Fill manually. Must describe the exact staging-training approval scope.",
        "approved_for_staging_training": "",
        "approved_dataset_or_staging_copy": EXPECTED_STAGING_COPY_REL,
        "approved_config_path": EXPECTED_CONFIG_REL,
        "approved_command_or_script": EXPECTED_SCRIPT_REL,
        "approved_expected_outputs": "",
        "approval_limitations": "",
        "explicit_human_confirmation": "Write a clear yes/no human confirmation from the approver.",
        "signature_or_confirmation_text": "",
    }


def build_validated_training_scope_markdown(
    *,
    all_passed: bool,
    approval_path: Path,
    approved_for_staging_training: bool,
    approved_dataset: str,
    approved_config: str,
    approved_command: str,
    expected_dataset: str,
    expected_yaml: str,
    expected_config: str,
    expected_script: str,
    expected_command: str,
    rejection_rows: list[dict[str, str]],
) -> str:
    lines = [
        "# Phase 11E-real Validated Training Scope",
        "",
        f"- approval_file_checked = {approval_path.as_posix()}",
        f"- real_human_approval_validated = {str(all_passed).lower()}",
        f"- training_allowed_after_phase11e = {str(all_passed).lower()}",
        f"- original_dataset_mutation_allowed = false",
        "",
    ]
    if all_passed:
        lines.extend(
            [
                "## Validated scope",
                f"- approved_for_staging_training = {str(approved_for_staging_training).lower()}",
                f"- approved_dataset_or_staging_copy = {approved_dataset}",
                f"- approved_config_path = {approved_config}",
                f"- approved_command_or_script = {approved_command}",
                "",
                "## Expected approved scope anchors",
                f"- expected staging copy = {expected_dataset}",
                f"- expected staging yaml = {expected_yaml}",
                f"- expected config = {expected_config}",
                f"- expected execution script = {expected_script}",
                f"- expected command preview = {expected_command}",
                "",
                "## Phase boundary",
                "- Phase 11E-real still does not execute training. It only validates that a later training-preparation phase may be created.",
            ]
        )
    else:
        lines.extend(
            [
                "## Validation result",
                "- The candidate approval file was rejected or remains incomplete.",
                "",
                "## Expected scope anchors for a later successful validation",
                f"- expected staging copy = {expected_dataset}",
                f"- expected staging yaml = {expected_yaml}",
                f"- expected config = {expected_config}",
                f"- expected execution script = {expected_script}",
                "",
                "## Rejection reasons",
            ]
        )
        for row in rejection_rows:
            lines.append(f"- {row['reason_id']}: {row['reason']}")
    return "\n".join(lines)


def build_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 11E-real Approval Validation",
            "",
            "This directory stores the content-validation results for the real human training approval gate.",
            "",
            "## Outputs",
            "",
            "- `phase11e_real_approval_validation_summary.json`",
            "- `phase11e_real_approval_validation_report.csv`",
            "- `phase11e_real_approval_acceptance_criteria.csv`",
            "- `phase11e_real_approval_rejection_reasons.csv`",
            "- `phase11e_validated_training_scope.md`",
            "- `phase11e_non_mutation_manifest.json`",
            "- `README.md`",
            "",
            "## Final status",
            "",
            f"- status = {summary['status']}",
            f"- phase11d_validated = {str(summary['phase11d_validated']).lower()}",
            f"- real_human_approval_file_checked = {summary['real_human_approval_file_checked']}",
            f"- real_human_approval_validated = {str(summary['real_human_approval_validated']).lower()}",
            f"- training_allowed_after_phase11e = {str(summary['training_allowed_after_phase11e']).lower()}",
            f"- historical_prior_gate_blocked = {str(summary['historical_prior_gate_blocked']).lower()}",
            f"- original_dataset_mutation_allowed = {str(summary['original_dataset_mutation_allowed']).lower()}",
            f"- next_allowed_step = {summary['next_allowed_step']}",
        ]
    )


def check_row(check_name: str, passed: bool, expected: str, observed: str, notes: str) -> dict[str, str]:
    return {
        "check_name": check_name,
        "passed": str(bool(passed)).lower(),
        "expected": expected,
        "observed": observed,
        "notes": notes,
    }


def rejection_reason(reason_id: str, reason: str, observed_value: str) -> dict[str, str]:
    return {
        "reason_id": reason_id,
        "reason": reason,
        "observed_value": observed_value,
    }


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_path_string(value: str, repo_root: Path | None = None) -> str:
    text = value.strip()
    if repo_root is not None and text.startswith(repo_root.as_posix()):
        return Path(text).relative_to(repo_root).as_posix()
    return text


def normalize_command_string(value: str) -> str:
    return " ".join(value.strip().split())


def is_blank(value: str) -> bool:
    return not value or not value.strip()


def looks_like_placeholder(value: str) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return True
    if normalized in PLACEHOLDER_TOKENS:
        return True
    return any(token in normalized for token in ("fill manually", "must describe", "write a clear", "placeholder", "example", "fixture"))


def normalize_yes_no(value: str) -> bool:
    normalized = normalize_text(value)
    return normalized in {"true", "yes", "approved", "approve", "y"}


def contains_affirmation_phrase(value: str) -> bool:
    normalized = normalize_text(value)
    return any(token in normalized for token in ("i approve", "explicitly approve", "approved by me", "human approved", "yes"))


def scope_blocks_original_mutation(value: str) -> bool:
    normalized = normalize_text(value)
    mutation_terms = ("original dataset", "original dataset labels", "data/yolo_format", "original labels")
    if not any(term in normalized for term in mutation_terms):
        return True
    negative_patterns = (
        "must not be mutated",
        "must not mutate",
        "no original dataset mutation",
        "original dataset must not be mutated",
        "original dataset labels must not be mutated",
        "do not mutate",
        "not be mutated",
        "no mutation",
    )
    return any(pattern in normalized for pattern in negative_patterns)


def write_outputs(*, output_dir: Path, results: dict[str, Any]) -> None:
    summary = results["summary"]
    write_json(output_dir / "phase11e_real_approval_validation_summary.json", summary)
    write_csv(
        output_dir / "phase11e_real_approval_validation_report.csv",
        results["report_rows"] + results["criteria_rows"],
        ["check_name", "passed", "expected", "observed", "notes"],
    )
    write_csv(
        output_dir / "phase11e_real_approval_acceptance_criteria.csv",
        results["criteria_rows"],
        ["check_name", "passed", "expected", "observed", "notes"],
    )
    write_csv(
        output_dir / "phase11e_real_approval_rejection_reasons.csv",
        results["rejection_rows"],
        ["reason_id", "reason", "observed_value"],
    )
    write_text(output_dir / "phase11e_validated_training_scope.md", results["validated_training_scope_md"])
    write_json(output_dir / "phase11e_non_mutation_manifest.json", results["non_mutation_manifest"])
    write_text(output_dir / "README.md", build_readme(summary))


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json_required(path, path.as_posix())


def read_json_required(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Required JSON missing: {label}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse {label}: {exc}") from exc


def read_text_required(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Required text file missing: {path}") from exc


def read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


if __name__ == "__main__":
    main()
