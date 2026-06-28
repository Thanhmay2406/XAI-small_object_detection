"""Prepare the Phase 11M.0 approved test evaluation package without executing evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11M.0"
STATUS_PASS = "phase11m0_approved_test_evaluation_prepare_only_passed"
STATUS_FAIL = "phase11m0_approved_test_evaluation_prepare_only_failed"
NEXT_PASS = "collect_phase11m1_explicit_evaluation_execution_approval_or_execute_with_flag"
NEXT_FAIL = "inspect_phase11l_and_dataset_yaml_resolution_before_evaluation"
DEFAULT_PHASE11L_SUMMARY = (
    "artifacts/phase11l_training_output_integrity_and_provenance/phase11l_training_output_integrity_summary.json"
)
DEFAULT_OUTPUT_DIR = "artifacts/phase11m0_prepare_approved_test_evaluation_no_execution"
DEFAULT_KAGGLE_DATASET_YAML = (
    "/kaggle/input/datasets/thanhmay2406/phase11-staging-dataset-relabel-patch-chipped/"
    "staging_dataset_copy/staging_dataset_drill_bit_yolo.yaml"
)
DEFAULT_PROJECT_DIR = "/home/thanhmay/workspace/XAI-small_object_detection/phase11m_test_eval"
DEFAULT_RUN_NAME = "yolov8n_drill_bit_phase11m_test_eval"
APPROVAL_FIELDS = ["phase", "approval_item", "approved", "approver", "approval_timestamp", "notes"]
PREFLIGHT_FIELDS = ["check_name", "passed", "severity", "observed_value", "expected_value", "notes"]
COMMAND_REVIEW_FIELDS = ["field", "value", "locked", "notes"]
REQUIRED_PHASE11L_KEYS = {
    "phase",
    "status",
    "baseline_checkpoint_candidate_accepted",
    "accepted_checkpoint_role",
    "accepted_checkpoint_path",
    "best_epoch",
    "best_metric_map50_95",
    "provenance_caveat",
    "next_allowed_step",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Phase 11M.0 approved test evaluation package without execution."
    )
    parser.add_argument("--phase11l-summary", default=DEFAULT_PHASE11L_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_phase11m0_approved_test_evaluation_no_execution(
        phase11l_summary_path=resolve_repo_path(args.phase11l_summary),
        output_dir=resolve_repo_path(args.output_dir),
    )
    concise = {
        "phase": summary["phase"],
        "status": summary["status"],
        "dataset_yaml_path": summary["dataset_yaml_path"],
        "dataset_yaml_resolution_status": summary["dataset_yaml_resolution_status"],
        "evaluation_runtime_target": summary["evaluation_runtime_target"],
        "locked_eval_command_prepared": summary["locked_eval_command_prepared"],
        "approved_for_phase11m1_execution": summary["approved_for_phase11m1_execution"],
        "next_allowed_step": summary["next_allowed_step"],
    }
    print(json.dumps(concise, indent=2))


def prepare_phase11m0_approved_test_evaluation_no_execution(
    phase11l_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight_rows: list[dict[str, Any]] = []

    phase11l_summary = read_json_if_exists(phase11l_summary_path)
    phase11l_summary_available = phase11l_summary is not None
    add_check(
        preflight_rows,
        "phase11l_summary_exists",
        phase11l_summary_available,
        "error",
        str(phase11l_summary_path.exists()),
        "True",
        "Phase 11M.0 requires the Phase 11L summary JSON.",
    )

    phase11l_keys_ok = phase11l_summary is not None and REQUIRED_PHASE11L_KEYS.issubset(phase11l_summary)
    add_check(
        preflight_rows,
        "phase11l_required_keys_present",
        phase11l_keys_ok,
        "error",
        sorted(list(phase11l_summary.keys())) if phase11l_summary else [],
        sorted(REQUIRED_PHASE11L_KEYS),
        "Phase 11L summary must expose the required fields for evaluation preparation.",
    )

    phase11l_status = str(phase11l_summary.get("status", "")) if phase11l_summary else ""
    phase11l_passed = (
        phase11l_summary is not None
        and phase11l_status == "phase11l_training_output_integrity_and_provenance_passed"
        and phase11l_summary.get("baseline_checkpoint_candidate_accepted") is True
        and phase11l_summary.get("accepted_checkpoint_role") == "best_pt_for_phase11m_prepare_only_evaluation"
        and phase11l_summary.get("next_allowed_step") == "phase11m0_prepare_approved_test_evaluation_no_execution"
    )
    add_check(
        preflight_rows,
        "phase11l_passed_gate",
        phase11l_passed,
        "error",
        {
            "status": phase11l_status,
            "baseline_checkpoint_candidate_accepted": phase11l_summary.get("baseline_checkpoint_candidate_accepted") if phase11l_summary else None,
            "accepted_checkpoint_role": phase11l_summary.get("accepted_checkpoint_role") if phase11l_summary else None,
            "next_allowed_step": phase11l_summary.get("next_allowed_step") if phase11l_summary else None,
        },
        {
            "status": "phase11l_training_output_integrity_and_provenance_passed",
            "baseline_checkpoint_candidate_accepted": True,
            "accepted_checkpoint_role": "best_pt_for_phase11m_prepare_only_evaluation",
            "next_allowed_step": "phase11m0_prepare_approved_test_evaluation_no_execution",
        },
        "Phase 11L must pass before any test-evaluation package is prepared.",
    )

    accepted_checkpoint_role = str(phase11l_summary.get("accepted_checkpoint_role", "")) if phase11l_summary else ""
    accepted_checkpoint_path = resolve_string_path(
        str(phase11l_summary.get("accepted_checkpoint_path", "")) if phase11l_summary else ""
    )
    accepted_checkpoint_exists = accepted_checkpoint_path is not None and accepted_checkpoint_path.exists()
    accepted_checkpoint_non_empty = accepted_checkpoint_exists and accepted_checkpoint_path.stat().st_size > 0
    add_check(
        preflight_rows,
        "accepted_checkpoint_metadata_present",
        bool(accepted_checkpoint_role and accepted_checkpoint_path),
        "error",
        {"role": accepted_checkpoint_role, "path": str(accepted_checkpoint_path) if accepted_checkpoint_path else ""},
        "non-empty role and path",
        "Phase 11L must provide the accepted checkpoint candidate metadata.",
    )
    add_check(
        preflight_rows,
        "accepted_checkpoint_exists",
        accepted_checkpoint_exists,
        "error",
        str(accepted_checkpoint_exists),
        "True",
        "The accepted checkpoint path must exist.",
    )
    add_check(
        preflight_rows,
        "accepted_checkpoint_non_empty",
        accepted_checkpoint_non_empty,
        "error",
        accepted_checkpoint_path.stat().st_size if accepted_checkpoint_exists else 0,
        "> 0",
        "The accepted checkpoint file must be non-empty.",
    )

    dataset_resolution = resolve_dataset_yaml(phase11l_summary, accepted_checkpoint_path)
    dataset_yaml_path = dataset_resolution["dataset_yaml_path"]
    add_check(
        preflight_rows,
        "dataset_yaml_resolved",
        dataset_resolution["dataset_yaml_resolved"],
        "error",
        dataset_yaml_path,
        "resolvable dataset yaml path",
        "Phase 11M.0 requires a resolved dataset YAML path for the locked command.",
    )
    add_check(
        preflight_rows,
        "dataset_yaml_resolution_status_recorded",
        bool(dataset_resolution["dataset_yaml_resolution_status"]),
        "error",
        dataset_resolution["dataset_yaml_resolution_status"],
        "non-empty resolution status",
        "Dataset YAML resolution status must be explicit.",
    )
    add_check(
        preflight_rows,
        "dataset_yaml_local_candidate_availability",
        True,
        "info",
        {
            "dataset_yaml_path": dataset_yaml_path,
            "exists_locally": dataset_resolution["dataset_yaml_exists_locally"],
            "ready_for_local_execution_candidate": dataset_resolution["ready_for_local_execution_candidate"],
            "ready_for_kaggle_execution_candidate": dataset_resolution["ready_for_kaggle_execution_candidate"],
        },
        "descriptive only",
        "A Kaggle-only dataset path is acceptable in prepare-only mode if it is clearly recorded.",
    )

    locked_command = build_locked_command(
        checkpoint_path=str(accepted_checkpoint_path) if accepted_checkpoint_path else "",
        dataset_yaml_path=dataset_yaml_path,
    )
    locked_command_prepared = bool(locked_command["command"])
    add_check(
        preflight_rows,
        "locked_eval_command_prepared",
        locked_command_prepared,
        "error",
        locked_command["command"],
        "non-empty locked yolo detect val command",
        "Phase 11M.0 prepares the command only and must not execute it.",
    )

    approval_rows = build_approval_template_rows()
    non_execution_manifest = build_non_execution_manifest()
    expected_outputs_manifest = build_expected_outputs_manifest()
    command_review_rows = build_command_review_rows(locked_command, dataset_resolution)

    preflight_passed = all(row["passed"] for row in preflight_rows if row["severity"] == "error")
    status = STATUS_PASS if preflight_passed else STATUS_FAIL
    next_allowed_step = NEXT_PASS if preflight_passed else NEXT_FAIL

    summary = {
        "phase": PHASE,
        "status": status,
        "phase11l_summary_available": phase11l_summary_available,
        "phase11l_status": phase11l_status,
        "phase11l_passed": phase11l_passed,
        "accepted_checkpoint_role": accepted_checkpoint_role,
        "accepted_checkpoint_path": str(accepted_checkpoint_path) if accepted_checkpoint_path else "",
        "accepted_checkpoint_exists": accepted_checkpoint_exists,
        "accepted_checkpoint_non_empty": accepted_checkpoint_non_empty,
        "checkpoint_loaded": False,
        "checkpoint_copied": False,
        "dataset_yaml_path": dataset_yaml_path,
        "dataset_yaml_resolution_status": dataset_resolution["dataset_yaml_resolution_status"],
        "evaluation_runtime_target": dataset_resolution["evaluation_runtime_target"],
        "ready_for_local_execution_candidate": dataset_resolution["ready_for_local_execution_candidate"],
        "ready_for_local_execution": False,
        "ready_for_kaggle_execution_candidate": dataset_resolution["ready_for_kaggle_execution_candidate"],
        "locked_eval_command_prepared": locked_command_prepared,
        "locked_eval_command_executed": False,
        "locked_eval_command_path": str(output_dir / "phase11m0_locked_test_evaluation_command.sh"),
        "locked_eval_command_json_path": str(output_dir / "phase11m0_locked_test_evaluation_command.json"),
        "approval_template_created": True,
        "approved_for_phase11m1_execution": False,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "prediction_executed": False,
        "export_executed": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "weights_modified": False,
        "weights_copied_to_artifacts": False,
        "ultralytics_eval_imported": False,
        "baseline_checkpoint_candidate_accepted_from_phase11l": (
            phase11l_summary.get("baseline_checkpoint_candidate_accepted") if phase11l_summary else False
        ),
        "best_epoch_from_phase11l": phase11l_summary.get("best_epoch") if phase11l_summary else None,
        "best_metric_map50_95_from_phase11l": (
            phase11l_summary.get("best_metric_map50_95") if phase11l_summary else None
        ),
        "provenance_caveat_carried_forward": (
            phase11l_summary.get("provenance_caveat") if phase11l_summary else ""
        ),
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": utc_now(),
    }

    dataset_resolution_output = {
        **dataset_resolution,
        "accepted_checkpoint_path": summary["accepted_checkpoint_path"],
        "phase11l_summary_path": str(phase11l_summary_path),
        "generated_at_utc": summary["generated_at_utc"],
    }
    locked_command_json = {
        "phase": PHASE,
        "command": locked_command["command"],
        "command_tokens": locked_command["command_tokens"],
        "commented_command": locked_command["commented_command"],
        "prepared_only": True,
        "executed": False,
        "dataset_yaml_path": dataset_yaml_path,
        "accepted_checkpoint_path": summary["accepted_checkpoint_path"],
        "project_dir": DEFAULT_PROJECT_DIR,
        "name": DEFAULT_RUN_NAME,
    }

    write_json(summary, output_dir / "phase11m0_prepare_test_evaluation_summary.json")
    (output_dir / "phase11m0_locked_test_evaluation_command.sh").write_text(
        build_shell_script(locked_command["commented_command"]), encoding="utf-8"
    )
    write_json(locked_command_json, output_dir / "phase11m0_locked_test_evaluation_command.json")
    write_csv(preflight_rows, output_dir / "phase11m0_preflight_checks.csv", PREFLIGHT_FIELDS)
    write_json(dataset_resolution_output, output_dir / "phase11m0_dataset_yaml_resolution.json")
    write_csv(approval_rows, output_dir / "phase11m0_evaluation_approval_template.csv", APPROVAL_FIELDS)
    write_json(non_execution_manifest, output_dir / "phase11m0_non_execution_manifest.json")
    write_csv(command_review_rows, output_dir / "phase11m0_command_review.csv", COMMAND_REVIEW_FIELDS)
    write_json(expected_outputs_manifest, output_dir / "phase11m0_expected_outputs_manifest.json")
    (output_dir / "README.md").write_text(build_readme(summary), encoding="utf-8")
    return summary


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_string_path(path_str: str) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_simple_yaml(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parsed[key.strip()] = value.strip().strip("'\"")
    return parsed


def resolve_dataset_yaml(phase11l_summary: dict[str, Any] | None, accepted_checkpoint_path: Path | None) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    phase11k_summary_path_str = ""
    if phase11l_summary:
        phase11k_summary_path_str = (
            phase11l_summary.get("validation_artifacts", {}).get("summary_json", "")
            if isinstance(phase11l_summary.get("validation_artifacts"), dict)
            else ""
        )
    phase11k_summary = read_json_if_exists(resolve_string_path(phase11k_summary_path_str)) if phase11k_summary_path_str else None
    if phase11k_summary:
        for key in ("dataset_yaml_path", "yaml_path"):
            value = str(phase11k_summary.get(key, "")).strip()
            if value:
                candidates.append(
                    {
                        "source": f"phase11k_summary.{key}",
                        "path": value,
                        "exists_locally": path_exists_locally(value),
                        "matches_phase11_staging": "phase11-staging-dataset" in value or "staging_dataset_drill_bit_yolo.yaml" in value,
                    }
                )

    if accepted_checkpoint_path is not None:
        args_yaml_path = accepted_checkpoint_path.parent.parent / "args.yaml"
        args_yaml = parse_simple_yaml(args_yaml_path)
        data_value = args_yaml.get("data", "")
        if data_value:
            candidates.append(
                {
                    "source": f"args_yaml:{args_yaml_path}",
                    "path": data_value,
                    "exists_locally": path_exists_locally(data_value),
                    "matches_phase11_staging": "phase11-staging-dataset" in data_value or "staging_dataset_drill_bit_yolo.yaml" in data_value,
                }
            )

    candidates.append(
        {
            "source": "default_prior_locked_kaggle_dataset_yaml_path",
            "path": DEFAULT_KAGGLE_DATASET_YAML,
            "exists_locally": path_exists_locally(DEFAULT_KAGGLE_DATASET_YAML),
            "matches_phase11_staging": True,
        }
    )

    repo_local_candidate = PROJECT_ROOT / "artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml"
    candidates.append(
        {
            "source": f"repo_local_candidate:{repo_local_candidate}",
            "path": str(repo_local_candidate),
            "exists_locally": repo_local_candidate.exists(),
            "matches_phase11_staging": repo_local_candidate.exists(),
        }
    )

    chosen = None
    for candidate in candidates:
        if candidate["path"] and candidate["matches_phase11_staging"] and candidate["exists_locally"]:
            chosen = candidate
            break
    if chosen is None:
        for candidate in candidates:
            if candidate["path"] and candidate["matches_phase11_staging"]:
                chosen = candidate
                break

    if chosen is None:
        return {
            "dataset_yaml_resolved": False,
            "dataset_yaml_path": "",
            "dataset_yaml_resolution_status": "unresolved_no_matching_dataset_yaml",
            "evaluation_runtime_target": "unknown",
            "ready_for_local_execution_candidate": False,
            "ready_for_kaggle_execution_candidate": False,
            "dataset_yaml_exists_locally": False,
            "candidates": candidates,
        }

    chosen_path = chosen["path"]
    chosen_exists_locally = bool(chosen["exists_locally"])
    if chosen_exists_locally:
        status = "resolved_local_existing_path"
        runtime_target = "local_or_environment_with_dataset_available"
        ready_local = True
        ready_kaggle = False
    elif chosen_path.startswith("/kaggle/"):
        status = "resolved_kaggle_path_not_locally_available"
        runtime_target = "kaggle_or_environment_with_dataset_mounted"
        ready_local = False
        ready_kaggle = True
    else:
        status = "resolved_nonlocal_path_not_available"
        runtime_target = "environment_specific_manual_resolution_required"
        ready_local = False
        ready_kaggle = False

    return {
        "dataset_yaml_resolved": True,
        "dataset_yaml_path": chosen_path,
        "dataset_yaml_resolution_status": status,
        "evaluation_runtime_target": runtime_target,
        "ready_for_local_execution_candidate": ready_local,
        "ready_for_kaggle_execution_candidate": ready_kaggle,
        "dataset_yaml_exists_locally": chosen_exists_locally,
        "chosen_source": chosen["source"],
        "candidates": candidates,
    }


def path_exists_locally(path_str: str) -> bool:
    path = Path(path_str)
    if path.is_absolute():
        return path.exists()
    return (PROJECT_ROOT / path).exists()


def build_locked_command(checkpoint_path: str, dataset_yaml_path: str) -> dict[str, Any]:
    tokens = [
        "yolo",
        "detect",
        "val",
        f"model={checkpoint_path}",
        f"data={dataset_yaml_path}",
        "split=test",
        "imgsz=640",
        "batch=16",
        f"project={DEFAULT_PROJECT_DIR}",
        f"name={DEFAULT_RUN_NAME}",
        "save_json=True",
        "save_conf=True",
        "plots=True",
    ]
    command = shlex.join(tokens)
    commented_command = "# " + command
    return {"command": command, "command_tokens": tokens, "commented_command": commented_command}


def build_shell_script(commented_command: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'echo "This command file is a prepare-only artifact from Phase 11M.0."',
            'echo "Do not execute it directly unless a later approved execution phase explicitly allows it."',
            "exit 1",
            "",
            "# Locked command below for review only:",
            commented_command,
            "",
        ]
    )


def build_approval_template_rows() -> list[dict[str, str]]:
    items = [
        "execute_test_evaluation_with_best_pt",
        "use_phase11l_baseline_checkpoint_candidate",
        "use_resolved_dataset_yaml",
        "write_evaluation_outputs_to_phase11m_test_eval",
        "confirm_no_training_or_dataset_mutation",
    ]
    return [
        {
            "phase": PHASE,
            "approval_item": item,
            "approved": "false",
            "approver": "",
            "approval_timestamp": "",
            "notes": "Explicit approval required before any later execution phase.",
        }
        for item in items
    ]


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
        "checkpoint_loaded": False,
        "ultralytics_training_imported": False,
        "ultralytics_eval_imported": False,
        "subprocess_eval_command_executed": False,
        "phase11m_test_eval_outputs_created": False,
    }


def build_expected_outputs_manifest() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "prepare_only": True,
        "expected_future_execution_outputs": [
            f"{DEFAULT_PROJECT_DIR}/{DEFAULT_RUN_NAME}/",
            f"{DEFAULT_PROJECT_DIR}/{DEFAULT_RUN_NAME}/args.yaml",
            f"{DEFAULT_PROJECT_DIR}/{DEFAULT_RUN_NAME}/results.csv",
            f"{DEFAULT_PROJECT_DIR}/{DEFAULT_RUN_NAME}/confusion_matrix.png",
            f"{DEFAULT_PROJECT_DIR}/{DEFAULT_RUN_NAME}/predictions.json",
        ],
        "note": "These outputs are expected only from a later explicit execution phase and are not created by Phase 11M.0.",
    }


def build_command_review_rows(locked_command: dict[str, Any], dataset_resolution: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "field": "command",
            "value": locked_command["command"],
            "locked": "true",
            "notes": "Prepared for review only; Phase 11M.0 does not execute it.",
        },
        {
            "field": "dataset_yaml_resolution_status",
            "value": dataset_resolution["dataset_yaml_resolution_status"],
            "locked": "true",
            "notes": "Runtime expectations depend on whether the dataset path exists locally or only in Kaggle.",
        },
        {
            "field": "evaluation_runtime_target",
            "value": dataset_resolution["evaluation_runtime_target"],
            "locked": "true",
            "notes": "Recorded for later execution-phase routing only.",
        },
    ]


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


def build_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 11M.0 Prepare Approved Test Evaluation No Execution",
            "",
            "Phase 11M.0 is a strict prepare-only and non-execution phase.",
            "",
            f"- status = `{summary['status']}`",
            f"- phase11l_status = `{summary['phase11l_status']}`",
            f"- accepted_checkpoint_path = `{summary['accepted_checkpoint_path']}`",
            f"- dataset_yaml_path = `{summary['dataset_yaml_path']}`",
            f"- dataset_yaml_resolution_status = `{summary['dataset_yaml_resolution_status']}`",
            f"- evaluation_runtime_target = `{summary['evaluation_runtime_target']}`",
            f"- locked_eval_command_prepared = `{summary['locked_eval_command_prepared']}`",
            f"- approved_for_phase11m1_execution = `{summary['approved_for_phase11m1_execution']}`",
            f"- next_allowed_step = `{summary['next_allowed_step']}`",
            "",
            "This phase does not run evaluation, inference, prediction, export, training, dataset mutation, label mutation, checkpoint loading, or weight copying.",
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
