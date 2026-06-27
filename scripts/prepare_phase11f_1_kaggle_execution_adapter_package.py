"""Prepare the Phase 11F.1 Kaggle execution adapter package without executing training."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "11F.1"
STATUS = "phase11f_1_kaggle_execution_adapter_package_prepared"
NEXT_ALLOWED_STEP = "phase11g_execute_approved_staging_training_on_kaggle"
REPO_ROOT_REQUIRED_DIRS = ("artifacts", "docs", "scripts", "src")
OUTPUT_DIR_DEFAULT = "artifacts/phase11f_1_kaggle_execution_adapter_package"
PHASE11F_SUMMARY_REL = (
    "artifacts/phase11f_approved_staging_training_execution_package/phase11f_execution_package_summary.json"
)
PHASE11F_COMMAND_REL = (
    "artifacts/phase11f_approved_staging_training_execution_package/phase11f_training_command.sh"
)
EXPECTED_PHASE11F_STATUS = "phase11f_approved_staging_training_execution_package_prepared"
EXPECTED_PHASE11F_NEXT = "phase11g_execute_approved_staging_training"
KAGGLE_REPO_ROOT_DEFAULT = "/kaggle/working/XAI-small_object_detection"
KAGGLE_PROJECT_DIR_DEFAULT = "/kaggle/working/experiments/phase11g_staging_training_chipped"
KAGGLE_TRAINING_NAME = "yolov8n_staging_chipped_phase11g"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Phase 11F.1 Kaggle execution adapter package without training."
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

    phase11f_summary = read_json_required(repo_root / PHASE11F_SUMMARY_REL, "Phase 11F summary")
    phase11f_command = read_text_required(repo_root / PHASE11F_COMMAND_REL, "Phase 11F training command")
    validate_phase11f_inputs(phase11f_summary)

    package = build_adapter_package(
        repo_root=repo_root,
        output_dir=output_dir,
        phase11f_summary=phase11f_summary,
        phase11f_command=phase11f_command,
    )
    write_outputs(output_dir, package)
    print(json.dumps(package["summary"], indent=2))


def ensure_repo_root(repo_root: Path) -> None:
    missing = [name for name in REPO_ROOT_REQUIRED_DIRS if not (repo_root / name).exists()]
    if missing:
        raise SystemExit(f"Repo root validation failed; missing required paths: {', '.join(missing)}")


def validate_phase11f_inputs(summary: dict[str, Any]) -> None:
    failures: list[str] = []
    if summary.get("status") != EXPECTED_PHASE11F_STATUS:
        failures.append(f"Phase 11F status must be '{EXPECTED_PHASE11F_STATUS}'.")
    if summary.get("ready_for_approved_training_execution") is not True:
        failures.append("Phase 11F requires ready_for_approved_training_execution == true.")
    if summary.get("next_allowed_step") != EXPECTED_PHASE11F_NEXT:
        failures.append(f"Phase 11F next_allowed_step must be '{EXPECTED_PHASE11F_NEXT}'.")
    if failures:
        raise SystemExit(" ".join(failures))


def build_adapter_package(
    repo_root: Path,
    output_dir: Path,
    phase11f_summary: dict[str, Any],
    phase11f_command: str,
) -> dict[str, Any]:
    local_command = extract_training_command(phase11f_command)
    parsed_command = parse_training_command(local_command)
    validate_training_semantics(parsed_command)

    relative_data_yaml = normalize_relpath(
        repo_root,
        phase11f_summary.get("approved_staging_dataset_yaml")
        or phase11f_summary.get("approved_staging_dataset_or_yaml"),
    )
    local_project_dir = str(parsed_command["project"])
    kaggle_command = build_kaggle_command(
        relative_data_yaml=relative_data_yaml,
        training_name=KAGGLE_TRAINING_NAME,
    )
    path_mapping_rows = build_path_mapping_rows(
        repo_root=repo_root,
        relative_data_yaml=relative_data_yaml,
        local_project_dir=local_project_dir,
    )
    guardrail_rows = build_guardrail_rows(phase11f_summary, parsed_command, kaggle_command)
    input_requirements = build_input_requirements(relative_data_yaml)
    non_execution_manifest = {
        "phase": PHASE,
        "status": STATUS,
        "prepare_only": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated_by_phase11f_1": False,
        "dataset_labels_mutated": False,
        "relabel_patch_applied": False,
        "architecture_changed": False,
        "loss_changed": False,
        "checkpoint_generated": False,
        "weights_generated": False,
        "phase11f_summary_mutated": False,
        "notes": "Phase 11F.1 only prepares a Kaggle adapter command package. It does not execute training, evaluation, or inference.",
    }
    summary = {
        "phase": "11F.1",
        "status": STATUS,
        "source_phase11f_status": phase11f_summary.get("status"),
        "kaggle_adapter_prepared": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated_by_phase11f_1": False,
        "command_semantics_preserved_from_phase11f": True,
        "local_absolute_paths_removed_or_parameterized": True,
        "ready_for_kaggle_phase11g_execution": True,
        "approved_staging_dataset_yaml_relative": relative_data_yaml,
        "kaggle_repo_root_default": KAGGLE_REPO_ROOT_DEFAULT,
        "kaggle_project_dir_default": KAGGLE_PROJECT_DIR_DEFAULT,
        "phase11f_summary_path": PHASE11F_SUMMARY_REL,
        "phase11f_training_command_path": PHASE11F_COMMAND_REL,
        "next_allowed_step": NEXT_ALLOWED_STEP,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    readme_text = build_readme(summary, relative_data_yaml)
    return {
        "summary": summary,
        "training_command_script": kaggle_command,
        "input_requirements": input_requirements,
        "path_mapping_rows": path_mapping_rows,
        "guardrail_rows": guardrail_rows,
        "non_execution_manifest": non_execution_manifest,
        "readme_text": readme_text,
    }


def extract_training_command(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped != "set -eu":
            return stripped
    raise SystemExit("Phase 11F training command file does not contain a usable prepared command.")


def parse_training_command(command: str) -> dict[str, str]:
    tokens = shlex.split(command)
    if len(tokens) < 4:
        raise SystemExit("Phase 11F training command is too short to parse.")
    parameters: dict[str, str] = {"binary": tokens[0], "mode_1": tokens[1], "mode_2": tokens[2]}
    for token in tokens[3:]:
        if "=" in token:
            key, value = token.split("=", 1)
            parameters[key] = value
        else:
            parameters.setdefault("extra_tokens", "")
            parameters["extra_tokens"] = (parameters["extra_tokens"] + " " + token).strip()
    return parameters


def validate_training_semantics(parsed_command: dict[str, str]) -> None:
    failures: list[str] = []
    if parsed_command.get("mode_1") != "detect" or parsed_command.get("mode_2") != "train":
        failures.append("Phase 11F command must remain 'yolo detect train'.")
    expected_pairs = {
        "model": "yolov8n.pt",
        "epochs": "100",
        "imgsz": "640",
        "batch": "16",
        "seed": "42",
    }
    for key, expected in expected_pairs.items():
        if parsed_command.get(key) != expected:
            failures.append(f"Phase 11F command must preserve {key}={expected}.")
    if not parsed_command.get("data"):
        failures.append("Phase 11F command must provide a data=... argument.")
    if failures:
        raise SystemExit(" ".join(failures))


def build_kaggle_command(relative_data_yaml: str, training_name: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# This command is prepared but not executed by Phase 11F.1.",
        "# Execute only in Phase 11G after confirming this Kaggle adapter package.",
        "# This training must use staging dataset/config only.",
        "# Original dataset mutation remains forbidden.",
        "",
        "set -eu",
        "",
        f'REPO_ROOT=${{REPO_ROOT:-{KAGGLE_REPO_ROOT_DEFAULT}}}',
        f'DATA_YAML=${{DATA_YAML:-$REPO_ROOT/{relative_data_yaml}}}',
        f'PROJECT_DIR=${{PROJECT_DIR:-{KAGGLE_PROJECT_DIR_DEFAULT}}}',
        "",
        (
            'yolo detect train data="$DATA_YAML" model=yolov8n.pt epochs=100 imgsz=640 '
            f'batch=16 seed=42 project="$PROJECT_DIR" name={training_name}'
        ),
        "",
    ]
    return "\n".join(lines)


def build_input_requirements(relative_data_yaml: str) -> str:
    lines = [
        "# Phase 11F.1 Kaggle Input Requirements",
        "",
        "Phase 11F.1 prepares a Kaggle execution adapter only. No training, evaluation, or inference is run here.",
        "",
        "## Required confirmations before Phase 11G on Kaggle",
        f"- Confirm `REPO_ROOT` points to the extracted or working-copy repo root. Default template: `{KAGGLE_REPO_ROOT_DEFAULT}`.",
        f"- Confirm `DATA_YAML` resolves to the staging dataset YAML equivalent on Kaggle. Default template: `$REPO_ROOT/{relative_data_yaml}`.",
        f"- Confirm `PROJECT_DIR` is writable under `/kaggle/working`. Default template: `{KAGGLE_PROJECT_DIR_DEFAULT}`.",
        "- Confirm the staging dataset YAML remains staging-only and does not redirect to the original dataset.",
        "- Confirm no command writes into `/kaggle/input`.",
        "- Confirm the approved training semantics remain unchanged: `model=yolov8n.pt`, `epochs=100`, `imgsz=640`, `batch=16`, `seed=42`.",
        "",
        "## Guardrails",
        "- Original dataset mutation remains forbidden.",
        "- Staging dataset mutation remains forbidden in Phase 11F.1.",
        "- Phase 11F summary is read-only input and is not modified by this phase.",
        "- If Kaggle repo root or dataset mount differs, override `REPO_ROOT`, `DATA_YAML`, or `PROJECT_DIR` explicitly before Phase 11G instead of editing the prepared semantics.",
    ]
    return "\n".join(lines) + "\n"


def build_path_mapping_rows(
    repo_root: Path,
    relative_data_yaml: str,
    local_project_dir: str,
) -> list[dict[str, str]]:
    return [
        mapping_row(
            str(repo_root),
            "${REPO_ROOT}",
            "repo_root",
            "Parameterize the local repo root for Kaggle working directory use.",
        ),
        mapping_row(
            str(repo_root / relative_data_yaml),
            "${DATA_YAML}",
            "staging_data_yaml",
            "Use the Kaggle-side staging dataset YAML while preserving staging-only semantics.",
        ),
        mapping_row(
            local_project_dir,
            "${PROJECT_DIR}",
            "project_dir",
            "Parameterize experiment output under /kaggle/working.",
        ),
    ]


def build_guardrail_rows(
    phase11f_summary: dict[str, Any],
    parsed_command: dict[str, str],
    kaggle_command: str,
) -> list[dict[str, str]]:
    return [
        row("phase11f_summary_exists", "Phase 11F summary exists", True, PHASE11F_SUMMARY_REL, "Required Phase 11F summary was loaded."),
        row(
            "phase11f_status_valid",
            "Phase 11F status valid",
            phase11f_summary.get("status") == EXPECTED_PHASE11F_STATUS,
            str(phase11f_summary.get("status")),
            "Phase 11F.1 only proceeds from the prepared Phase 11F package.",
        ),
        row(
            "phase11f_ready_for_execution",
            "Phase 11F ready for approved training execution",
            phase11f_summary.get("ready_for_approved_training_execution") is True,
            str(phase11f_summary.get("ready_for_approved_training_execution")),
            "Phase 11F.1 requires the Phase 11F package to be execution-ready.",
        ),
        row(
            "phase11f_next_step_valid",
            "Phase 11F next allowed step valid",
            phase11f_summary.get("next_allowed_step") == EXPECTED_PHASE11F_NEXT,
            str(phase11f_summary.get("next_allowed_step")),
            "Phase 11F must point to Phase 11G before Kaggle adaptation is prepared.",
        ),
        row("no_training_executed", "No training executed", True, "false", "Phase 11F.1 prepares a Kaggle adapter only."),
        row("no_evaluation_executed", "No evaluation executed", True, "false", "Phase 11F.1 does not evaluate anything."),
        row("no_inference_executed", "No inference executed", True, "false", "Phase 11F.1 does not run inference."),
        row("original_dataset_not_mutated", "Original dataset not mutated", True, "false", "Original dataset mutation remains forbidden."),
        row("staging_dataset_not_mutated", "Staging dataset not mutated by Phase 11F.1", True, "false", "Phase 11F.1 does not alter staging data."),
        row("phase11f_summary_not_mutated", "Phase 11F summary not mutated", True, "false", "Phase 11F is treated as read-only input."),
        row(
            "training_semantics_preserved",
            "Training semantics preserved from Phase 11F",
            True,
            (
                f"model={parsed_command.get('model')}; epochs={parsed_command.get('epochs')}; "
                f"imgsz={parsed_command.get('imgsz')}; batch={parsed_command.get('batch')}; seed={parsed_command.get('seed')}"
            ),
            "Core training arguments are preserved exactly from Phase 11F.",
        ),
        row(
            "kaggle_training_name_adapted",
            "Kaggle command uses Phase 11G execution name",
            f"name={KAGGLE_TRAINING_NAME}" in kaggle_command,
            KAGGLE_TRAINING_NAME,
            "Phase 11F.1 renames the run to the Phase 11G Kaggle execution target while preserving core training arguments.",
        ),
        row(
            "local_paths_parameterized",
            "Local absolute paths removed or parameterized",
            "${REPO_ROOT:-" in kaggle_command and "${DATA_YAML:-" in kaggle_command and "${PROJECT_DIR:-" in kaggle_command,
            "REPO_ROOT/DATA_YAML/PROJECT_DIR variables present",
            "Kaggle adapter command avoids hard-coded local absolute paths.",
        ),
    ]


def build_readme(summary: dict[str, Any], relative_data_yaml: str) -> str:
    lines = [
        "# Phase 11F.1 Kaggle Execution Adapter Package",
        "",
        "Phase 11F.1 prepares a Kaggle adapter version of the Phase 11F training command without executing it.",
        "",
        "## Outcome",
        f"- status = `{summary['status']}`",
        f"- source_phase11f_status = `{summary['source_phase11f_status']}`",
        f"- kaggle_adapter_prepared = `{summary['kaggle_adapter_prepared']}`",
        f"- training_executed = `{summary['training_executed']}`",
        f"- evaluation_executed = `{summary['evaluation_executed']}`",
        f"- inference_executed = `{summary['inference_executed']}`",
        f"- original_dataset_mutated = `{summary['original_dataset_mutated']}`",
        f"- staging_dataset_mutated_by_phase11f_1 = `{summary['staging_dataset_mutated_by_phase11f_1']}`",
        f"- next_allowed_step = `{summary['next_allowed_step']}`",
        "",
        "## Prepared Kaggle path defaults",
        f"- `REPO_ROOT={summary['kaggle_repo_root_default']}`",
        f"- `DATA_YAML=$REPO_ROOT/{relative_data_yaml}`",
        f"- `PROJECT_DIR={summary['kaggle_project_dir_default']}`",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, package: dict[str, Any]) -> None:
    write_json(output_dir / "phase11f_1_kaggle_adapter_summary.json", package["summary"])
    write_text(output_dir / "phase11f_1_kaggle_training_command.sh", package["training_command_script"])
    write_text(output_dir / "phase11f_1_kaggle_input_requirements.md", package["input_requirements"])
    write_csv(
        output_dir / "phase11f_1_path_mapping.csv",
        package["path_mapping_rows"],
        ["local_path", "kaggle_path_or_variable", "mapping_role", "notes"],
    )
    write_csv(
        output_dir / "phase11f_1_guardrail_checks.csv",
        package["guardrail_rows"],
        ["check_id", "check_name", "result", "observed_value", "notes"],
    )
    write_json(output_dir / "phase11f_1_non_execution_manifest.json", package["non_execution_manifest"])
    write_text(output_dir / "README.md", package["readme_text"])


def read_json_required(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing required {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text_required(path: Path, label: str) -> str:
    if not path.exists():
        raise SystemExit(f"Missing required {label}: {path}")
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


def row(check_id: str, check_name: str, passed: bool, observed_value: str, notes: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "check_name": check_name,
        "result": "pass" if passed else "fail",
        "observed_value": observed_value,
        "notes": notes,
    }


def mapping_row(local_path: str, kaggle_path_or_variable: str, mapping_role: str, notes: str) -> dict[str, str]:
    return {
        "local_path": local_path,
        "kaggle_path_or_variable": kaggle_path_or_variable,
        "mapping_role": mapping_role,
        "notes": notes,
    }


if __name__ == "__main__":
    main()
