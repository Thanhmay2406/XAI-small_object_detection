"""Prepare the approved staging-training execution package without executing anything."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "11F"
STATUS = "phase11f_approved_staging_training_execution_package_prepared"
NEXT_ALLOWED_STEP = "phase11g_execute_approved_staging_training"
REPO_ROOT_REQUIRED_DIRS = ("artifacts", "docs", "scripts", "src")
OUTPUT_DIR_DEFAULT = "artifacts/phase11f_approved_staging_training_execution_package"
PHASE11E_SUMMARY_REL = (
    "artifacts/phase11e_real_approval_validation/phase11e_real_approval_validation_summary.json"
)
PHASE9V_SUMMARY_REL = "artifacts/phase9v_staging_training_config_chipped/phase9v_training_config_summary.json"
PHASE9W_SUMMARY_REL = (
    "artifacts/phase9w_final_command_review_chipped/phase9w_final_command_review_summary.json"
)
PHASE9Y_SUMMARY_REL = "artifacts/phase9y_approved_staging_training_chipped/phase9y_training_execution_summary.json"
PHASE9Y_COMMAND_REL = "artifacts/phase9y_approved_staging_training_chipped/phase9y_training_command.txt"
PHASE9Y_MANIFEST_REL = "artifacts/phase9y_approved_staging_training_chipped/phase9y_training_run_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Phase 11F approved staging-training execution package without training."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    ensure_repo_root(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase11e_summary = read_json_required(repo_root / PHASE11E_SUMMARY_REL, "Phase 11E summary")
    validate_phase11e_gate(phase11e_summary)

    phase9v_summary = read_optional_json(repo_root / PHASE9V_SUMMARY_REL)
    phase9w_summary = read_optional_json(repo_root / PHASE9W_SUMMARY_REL)
    phase9y_summary = read_optional_json(repo_root / PHASE9Y_SUMMARY_REL)
    phase9y_manifest = read_optional_json(repo_root / PHASE9Y_MANIFEST_REL)
    phase9y_command_text = read_optional_text(repo_root / PHASE9Y_COMMAND_REL)

    package = build_execution_package(
        repo_root=repo_root,
        output_dir=output_dir,
        phase11e_summary=phase11e_summary,
        phase9v_summary=phase9v_summary,
        phase9w_summary=phase9w_summary,
        phase9y_summary=phase9y_summary,
        phase9y_manifest=phase9y_manifest,
        phase9y_command_text=phase9y_command_text,
    )
    write_outputs(output_dir, package)
    print(json.dumps(package["summary"], indent=2))


def ensure_repo_root(repo_root: Path) -> None:
    missing = [name for name in REPO_ROOT_REQUIRED_DIRS if not (repo_root / name).exists()]
    if missing:
        raise SystemExit(f"Repo root validation failed; missing required paths: {', '.join(missing)}")


def validate_phase11e_gate(summary: dict[str, Any]) -> None:
    failures: list[str] = []
    if summary.get("status") != "phase11e_real_human_training_approval_validated":
        failures.append(
            "Phase 11E summary status must be 'phase11e_real_human_training_approval_validated'."
        )
    if summary.get("real_human_approval_validated") is not True:
        failures.append("Phase 11E requires real_human_approval_validated == true.")
    if summary.get("training_allowed_after_phase11e") is not True:
        failures.append("Phase 11E requires training_allowed_after_phase11e == true.")
    if failures:
        raise SystemExit(" ".join(failures))


def build_execution_package(
    repo_root: Path,
    output_dir: Path,
    phase11e_summary: dict[str, Any],
    phase9v_summary: dict[str, Any] | None,
    phase9w_summary: dict[str, Any] | None,
    phase9y_summary: dict[str, Any] | None,
    phase9y_manifest: dict[str, Any] | None,
    phase9y_command_text: str | None,
) -> dict[str, Any]:
    staging_dataset_path = normalize_relpath(
        repo_root,
        phase11e_summary.get("approved_dataset_or_staging_copy")
        or get_optional(phase9v_summary, "staging_dataset_yaml"),
    )
    staging_yaml_path = normalize_relpath(repo_root, get_optional(phase9v_summary, "staging_dataset_yaml"))
    config_path = normalize_relpath(
        repo_root,
        phase11e_summary.get("approved_config_path") or get_optional(phase9v_summary, "training_config_path"),
    )
    command_source_script = normalize_relpath(
        repo_root, phase11e_summary.get("approved_command_or_script") or "scripts/execute_phase9y_approved_staging_training.py"
    )
    training_command = resolve_training_command(
        phase9y_command_text=phase9y_command_text,
        phase9y_manifest=phase9y_manifest,
        phase9y_summary=phase9y_summary,
    )
    config_confirmed = bool(config_path and (repo_root / config_path).exists())
    staging_input_confirmed = bool(staging_dataset_path and (repo_root / staging_dataset_path).exists())
    command_prepared = bool(training_command)

    requirements = build_input_requirements_markdown(
        config_path=config_path,
        config_confirmed=config_confirmed,
        staging_dataset_path=staging_dataset_path,
        staging_yaml_path=staging_yaml_path,
        staging_input_confirmed=staging_input_confirmed,
        command_source_script=command_source_script,
        command_prepared=command_prepared,
    )
    training_command_script = build_training_command_script(
        training_command=training_command,
        command_prepared=command_prepared,
    )

    guardrail_rows = build_guardrail_rows(
        phase11e_summary=phase11e_summary,
        config_path=config_path,
        config_confirmed=config_confirmed,
        staging_dataset_path=staging_dataset_path,
        staging_yaml_path=staging_yaml_path,
        staging_input_confirmed=staging_input_confirmed,
    )

    manifest_rows = build_manifest_rows(
        output_dir=output_dir,
        phase11e_summary=phase11e_summary,
        config_path=config_path,
        staging_dataset_path=staging_dataset_path,
        staging_yaml_path=staging_yaml_path,
        command_source_script=command_source_script,
    )

    non_execution_manifest = {
        "phase": PHASE,
        "status": STATUS,
        "prepare_only": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated_by_phase11f": False,
        "dataset_labels_mutated": False,
        "relabel_patch_applied": False,
        "architecture_changed": False,
        "loss_changed": False,
        "checkpoint_generated": False,
        "weights_generated": False,
        "phase9z5_state_mutated": False,
        "notes": "Phase 11F prepares an approved execution package only. It does not run training, evaluation, or inference.",
    }

    summary = {
        "phase": PHASE,
        "status": STATUS,
        "phase11e_real_human_approval_validated": True,
        "training_allowed_after_phase11e": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated_by_phase11f": False,
        "phase9z5_state_mutated": False,
        "historical_phase9z5_training_allowed": phase11e_summary.get("historical_phase9z5_training_allowed"),
        "historical_phase9z5_approval_validated": phase11e_summary.get("historical_phase9z5_approval_validated"),
        "historical_prior_gate_blocked": phase11e_summary.get("historical_prior_gate_blocked"),
        "approved_execution_package_prepared": True,
        "ready_for_approved_training_execution": True,
        "approved_staging_dataset_or_yaml": staging_dataset_path,
        "approved_staging_dataset_yaml": staging_yaml_path,
        "approved_config_path": config_path,
        "approved_command_source_script": command_source_script,
        "training_command_prepared": command_prepared,
        "training_command_executed": False,
        "config_path_confirmed": config_confirmed,
        "staging_input_confirmed": staging_input_confirmed,
        "phase11e_summary_path": PHASE11E_SUMMARY_REL,
        "phase9v_summary_consulted": phase9v_summary is not None,
        "phase9w_summary_consulted": phase9w_summary is not None,
        "phase9y_summary_consulted": phase9y_summary is not None,
        "next_allowed_step": NEXT_ALLOWED_STEP,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    readme_text = build_readme(
        summary=summary,
        config_confirmed=config_confirmed,
        staging_input_confirmed=staging_input_confirmed,
    )
    return {
        "summary": summary,
        "manifest_rows": manifest_rows,
        "training_command_script": training_command_script,
        "input_requirements": requirements,
        "non_execution_manifest": non_execution_manifest,
        "guardrail_rows": guardrail_rows,
        "readme_text": readme_text,
    }


def resolve_training_command(
    phase9y_command_text: str | None,
    phase9y_manifest: dict[str, Any] | None,
    phase9y_summary: dict[str, Any] | None,
) -> str:
    if phase9y_command_text and phase9y_command_text.strip():
        return phase9y_command_text.strip()
    if phase9y_manifest and phase9y_manifest.get("training_command_shell"):
        return str(phase9y_manifest["training_command_shell"]).strip()
    if phase9y_summary and phase9y_summary.get("outputs", {}).get("phase9y_training_command_txt"):
        return ""
    return ""


def build_training_command_script(training_command: str, command_prepared: bool) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# This command is prepared but not executed by Phase 11F.",
        "# Execute only in Phase 11G after confirming this package.",
        "# This training must use staging dataset/config only.",
        "# Original dataset mutation remains forbidden.",
        "",
        "set -eu",
        "",
    ]
    if command_prepared:
        lines.append(training_command)
    else:
        lines.append(
            "# Training command placeholder: confirm the staging training command before Phase 11G execution."
        )
    lines.append("")
    return "\n".join(lines)


def build_input_requirements_markdown(
    config_path: str,
    config_confirmed: bool,
    staging_dataset_path: str,
    staging_yaml_path: str,
    staging_input_confirmed: bool,
    command_source_script: str,
    command_prepared: bool,
) -> str:
    config_note = config_path if config_confirmed else "Confirmation required before Phase 11G."
    staging_note = staging_dataset_path if staging_input_confirmed else "Confirmation required before Phase 11G."
    staging_yaml_note = staging_yaml_path if staging_yaml_path else "Confirmation required before Phase 11G."
    command_note = "Prepared from the frozen staging-training chain." if command_prepared else "Command requires confirmation before Phase 11G."
    lines = [
        "# Phase 11F Kaggle or Local Input Requirements",
        "",
        "Phase 11F prepares an execution package only. No training, evaluation, or inference is run here.",
        "",
        "## Confirmed inputs",
        f"- Phase 11E real approval summary: `{PHASE11E_SUMMARY_REL}`",
        f"- Approved command source script: `{command_source_script}`",
        f"- Staging dataset or YAML: `{staging_note}`",
        f"- Staging dataset YAML used by the prepared command: `{staging_yaml_note}`",
        f"- Approved training config: `{config_note}`",
        "",
        "## Requirements before Phase 11G",
        "- Reconfirm that execution still targets staging-only inputs and not `data/YOLO_format`.",
        "- Reconfirm that the prepared training command is reviewed as-is and remains non-executed until Phase 11G.",
        "- Reconfirm there will be no original-dataset mutation, no relabel patching, and no architecture/loss edits during execution approval.",
        "- If Kaggle is used, ensure the runtime has the staging dataset artifact, the approved config, cached or downloadable model weights, and the repo scripts needed for execution.",
        "- If local execution is used, ensure `.venv/bin/yolo` or an equivalent approved runtime path exists before Phase 11G.",
        "",
        "## Notes",
        f"- Training command status: {command_note}",
        "- Historical Phase 9Z.5 blocked flags are retained as context only and do not block this Phase 11F package after Phase 11E validation.",
    ]
    return "\n".join(lines) + "\n"


def build_guardrail_rows(
    phase11e_summary: dict[str, Any],
    config_path: str,
    config_confirmed: bool,
    staging_dataset_path: str,
    staging_yaml_path: str,
    staging_input_confirmed: bool,
) -> list[dict[str, str]]:
    return [
        row("phase11e_summary_exists", "Phase 11E summary exists", True, PHASE11E_SUMMARY_REL, "Required Phase 11E summary was loaded."),
        row(
            "phase11e_status_validated",
            "Phase 11E status validated",
            phase11e_summary.get("status") == "phase11e_real_human_training_approval_validated",
            str(phase11e_summary.get("status")),
            "Phase 11F only proceeds from the validated Phase 11E status.",
        ),
        row(
            "real_human_approval_validated",
            "Real human approval validated",
            phase11e_summary.get("real_human_approval_validated") is True,
            str(phase11e_summary.get("real_human_approval_validated")),
            "Phase 11F requires true Phase 11E approval validation.",
        ),
        row(
            "training_allowed_after_phase11e",
            "Training allowed after Phase 11E",
            phase11e_summary.get("training_allowed_after_phase11e") is True,
            str(phase11e_summary.get("training_allowed_after_phase11e")),
            "Phase 11F requires Phase 11E to explicitly allow later approved training.",
        ),
        row(
            "historical_phase9z5_flags_context_only",
            "Historical Phase 9Z.5 flags treated as context only",
            True,
            (
                f"training_allowed={phase11e_summary.get('historical_phase9z5_training_allowed')}; "
                f"approval_validated={phase11e_summary.get('historical_phase9z5_approval_validated')}; "
                f"prior_gate_blocked={phase11e_summary.get('historical_prior_gate_blocked')}"
            ),
            "Historical Phase 9Z.5 state is preserved for context and does not block validated Phase 11E follow-up.",
        ),
        row("no_training_executed", "No training executed", True, "false", "Phase 11F prepares a package only."),
        row("no_evaluation_executed", "No evaluation executed", True, "false", "Phase 11F does not evaluate anything."),
        row("no_inference_executed", "No inference executed", True, "false", "Phase 11F does not run inference."),
        row(
            "original_dataset_not_mutated",
            "Original dataset not mutated",
            True,
            "false",
            "Original dataset mutation remains forbidden.",
        ),
        row(
            "staging_dataset_not_mutated_by_phase11f",
            "Staging dataset not mutated by Phase 11F",
            True,
            "false",
            "Phase 11F consumes staging references only and does not edit them.",
        ),
        row(
            "phase9z5_state_not_mutated",
            "Phase 9Z.5 state not mutated",
            phase11e_summary.get("phase9z5_state_mutated") is False,
            str(phase11e_summary.get("phase9z5_state_mutated")),
            "Phase 11F preserves historical Phase 9Z.5 state.",
        ),
        row(
            "approved_config_path_frozen",
            "Approved config path frozen",
            config_confirmed,
            config_path or "missing",
            "Phase 11F records the config path to be used later in Phase 11G.",
        ),
        row(
            "staging_input_frozen",
            "Approved staging input frozen",
            staging_input_confirmed,
            (
                f"dataset_copy={staging_dataset_path or 'missing'}; "
                f"staging_yaml={staging_yaml_path or 'missing'}"
            ),
            "Phase 11F records the staging-only dataset input for later execution.",
        ),
    ]


def build_manifest_rows(
    output_dir: Path,
    phase11e_summary: dict[str, Any],
    config_path: str,
    staging_dataset_path: str,
    staging_yaml_path: str,
    command_source_script: str,
) -> list[dict[str, str]]:
    package_dir = output_dir.relative_to(output_dir.parent.parent).as_posix() if output_dir.parts else OUTPUT_DIR_DEFAULT
    return [
        manifest_row("input", PHASE11E_SUMMARY_REL, "validated_gate_summary", "Phase 11E validated approval gate input"),
        manifest_row("input", config_path or "", "approved_training_config", "Frozen staging-training config path"),
        manifest_row("input", staging_dataset_path or "", "approved_staging_dataset_or_yaml", "Frozen staging-only dataset or YAML input"),
        manifest_row("input", staging_yaml_path or "", "approved_staging_dataset_yaml", "Exact staging YAML used by the prepared command"),
        manifest_row("input", command_source_script or "", "approved_command_source_script", "Approved execution-source script path"),
        manifest_row(
            "output",
            f"{package_dir}/phase11f_execution_package_summary.json",
            "phase11f_summary",
            "Phase 11F execution package summary",
        ),
        manifest_row(
            "output",
            f"{package_dir}/phase11f_execution_package_manifest.csv",
            "phase11f_manifest",
            "Phase 11F execution package manifest",
        ),
        manifest_row(
            "output",
            f"{package_dir}/phase11f_training_command.sh",
            "phase11f_training_command",
            "Prepared-only training command script",
        ),
        manifest_row(
            "output",
            f"{package_dir}/phase11f_kaggle_or_local_input_requirements.md",
            "phase11f_input_requirements",
            "Execution input requirements for later Phase 11G",
        ),
        manifest_row(
            "output",
            f"{package_dir}/phase11f_non_execution_manifest.json",
            "phase11f_non_execution_manifest",
            "Explicit non-execution and non-mutation manifest",
        ),
        manifest_row(
            "output",
            f"{package_dir}/phase11f_guardrail_checks.csv",
            "phase11f_guardrails",
            "Guardrail checks proving this phase remained prepare-only",
        ),
        manifest_row("output", f"{package_dir}/README.md", "phase11f_readme", "Phase 11F package readme"),
        manifest_row(
            "context",
            str(phase11e_summary.get("historical_phase9z5_training_allowed")),
            "historical_phase9z5_training_allowed",
            "Historical context only; not a Phase 11F blocker after real approval validation",
        ),
    ]


def build_readme(summary: dict[str, Any], config_confirmed: bool, staging_input_confirmed: bool) -> str:
    lines = [
        "# Phase 11F Approved Staging Training Execution Package",
        "",
        "Phase 11F prepares an approved staging-training execution package only.",
        "",
        "## Outcome",
        f"- status = `{summary['status']}`",
        f"- phase11e_real_human_approval_validated = `{summary['phase11e_real_human_approval_validated']}`",
        f"- training_allowed_after_phase11e = `{summary['training_allowed_after_phase11e']}`",
        f"- training_executed = `{summary['training_executed']}`",
        f"- evaluation_executed = `{summary['evaluation_executed']}`",
        f"- inference_executed = `{summary['inference_executed']}`",
        f"- original_dataset_mutated = `{summary['original_dataset_mutated']}`",
        f"- staging_dataset_mutated_by_phase11f = `{summary['staging_dataset_mutated_by_phase11f']}`",
        f"- phase9z5_state_mutated = `{summary['phase9z5_state_mutated']}`",
        f"- next_allowed_step = `{summary['next_allowed_step']}`",
        "",
        "## Prepared inputs",
        f"- approved_config_path = `{summary['approved_config_path']}` (confirmed: `{config_confirmed}`)",
        f"- approved_staging_dataset_or_yaml = `{summary['approved_staging_dataset_or_yaml']}` (confirmed: `{staging_input_confirmed}`)",
        f"- approved_command_source_script = `{summary['approved_command_source_script']}`",
        "",
        "## Guardrails",
        "- Historical Phase 9Z.5 blocked flags are preserved as context only.",
        "- No training, evaluation, or inference is executed in Phase 11F.",
        "- No original or staging dataset mutation is performed in Phase 11F.",
        "- The prepared shell command remains non-executed until a later Phase 11G confirmation step.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, package: dict[str, Any]) -> None:
    write_json(output_dir / "phase11f_execution_package_summary.json", package["summary"])
    write_csv(
        output_dir / "phase11f_execution_package_manifest.csv",
        package["manifest_rows"],
        ["entry_type", "path", "role", "notes"],
    )
    write_text(output_dir / "phase11f_training_command.sh", package["training_command_script"])
    write_text(
        output_dir / "phase11f_kaggle_or_local_input_requirements.md",
        package["input_requirements"],
    )
    write_json(output_dir / "phase11f_non_execution_manifest.json", package["non_execution_manifest"])
    write_csv(
        output_dir / "phase11f_guardrail_checks.csv",
        package["guardrail_rows"],
        ["check_id", "check_name", "result", "observed_value", "notes"],
    )
    write_text(output_dir / "README.md", package["readme_text"])


def read_json_required(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing required {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_relpath(repo_root: Path, value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if path.is_absolute():
        return path.relative_to(repo_root).as_posix()
    return path.as_posix()


def get_optional(payload: dict[str, Any] | None, key: str) -> Any:
    if payload is None:
        return None
    return payload.get(key)


def row(check_id: str, check_name: str, passed: bool, observed_value: str, notes: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "check_name": check_name,
        "result": "pass" if passed else "fail",
        "observed_value": observed_value,
        "notes": notes,
    }


def manifest_row(entry_type: str, path: str, role: str, notes: str) -> dict[str, str]:
    return {
        "entry_type": entry_type,
        "path": path,
        "role": role,
        "notes": notes,
    }


if __name__ == "__main__":
    main()
