"""Execute the Phase 11M.1 approved test evaluation only when explicitly allowed."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11M.1"
STATUS_BLOCKED_DEFAULT = "phase11m1_test_evaluation_blocked_missing_execute_or_approval"
STATUS_BLOCKED_PREFLIGHT = "phase11m1_test_evaluation_blocked_preflight_failed"
STATUS_EXECUTED = "phase11m1_test_evaluation_executed_successfully"
STATUS_FAILED = "phase11m1_test_evaluation_execution_failed"
NEXT_BLOCKED_DEFAULT = "provide_explicit_execute_flag_or_filled_phase11m1_approval_csv"
NEXT_BLOCKED_PREFLIGHT = "fix_runtime_model_or_dataset_yaml_path_before_evaluation"
NEXT_EXECUTED = "phase11n_collect_and_validate_test_evaluation_outputs"
NEXT_FAILED = "inspect_phase11m1_stdout_stderr_and_runtime_paths"
DEFAULT_PHASE11M0_SUMMARY = (
    "artifacts/phase11m0_prepare_approved_test_evaluation_no_execution/phase11m0_prepare_test_evaluation_summary.json"
)
DEFAULT_OUTPUT_DIR = "artifacts/phase11m1_approved_test_evaluation_execution"
DEFAULT_RUN_NAME = "yolov8n_drill_bit_phase11m_test_eval"
DEFAULT_REPO_PROJECT = "/home/thanhmay/workspace/XAI-small_object_detection/phase11m_test_eval"
APPROVAL_FIELDS = ["phase", "approval_item", "approved", "approver", "approval_timestamp", "notes"]
APPROVAL_ITEMS = [
    "execute_test_evaluation_with_phase11l_best_pt",
    "use_runtime_model_path",
    "use_runtime_dataset_yaml",
    "write_outputs_to_phase11m_test_eval",
    "confirm_no_training",
    "confirm_no_dataset_mutation",
    "confirm_no_label_mutation",
    "confirm_no_weight_copy_or_modification",
]
PREFLIGHT_FIELDS = ["check_name", "passed", "severity", "observed_value", "expected_value", "notes"]
OUTPUT_MANIFEST_FIELDS = ["relative_path", "exists", "is_dir", "size_bytes", "modified_time_utc", "note"]
REQUIRED_PHASE11M0_KEYS = {
    "phase",
    "status",
    "phase11l_passed",
    "accepted_checkpoint_path",
    "accepted_checkpoint_exists",
    "accepted_checkpoint_non_empty",
    "dataset_yaml_path",
    "dataset_yaml_resolution_status",
    "evaluation_runtime_target",
    "ready_for_kaggle_execution_candidate",
    "locked_eval_command_prepared",
    "approved_for_phase11m1_execution",
    "baseline_checkpoint_candidate_accepted_from_phase11l",
    "best_epoch_from_phase11l",
    "best_metric_map50_95_from_phase11l",
    "provenance_caveat_carried_forward",
    "next_allowed_step",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the Phase 11M.1 approved test evaluation only with explicit approval."
    )
    parser.add_argument("--phase11m0-summary", default=DEFAULT_PHASE11M0_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--approval-csv", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", default="test")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = execute_phase11m1_approved_test_evaluation(
        phase11m0_summary_path=resolve_repo_path(args.phase11m0_summary),
        output_dir=resolve_repo_path(args.output_dir),
        approval_csv_path=resolve_optional_repo_path(args.approval_csv),
        model_path_override=args.model_path,
        data_yaml_override=args.data_yaml,
        project_override=args.project,
        run_name=args.name,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        execute=args.execute,
    )
    concise = {
        "phase": summary["phase"],
        "status": summary["status"],
        "execution_requested": summary["execution_requested"],
        "execution_allowed": summary["execution_allowed"],
        "locked_runtime_command_executed": summary["locked_runtime_command_executed"],
        "runtime_target": summary["runtime_target"],
        "runtime_model_path": summary["runtime_model_path"],
        "runtime_data_yaml": summary["runtime_data_yaml"],
        "next_allowed_step": summary["next_allowed_step"],
    }
    print(json.dumps(concise, indent=2))


def execute_phase11m1_approved_test_evaluation(
    phase11m0_summary_path: Path,
    output_dir: Path,
    approval_csv_path: Path | None,
    model_path_override: str,
    data_yaml_override: str,
    project_override: str,
    run_name: str,
    imgsz: int,
    batch: int,
    split: str,
    execute: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path = output_dir / "phase11m1_stdout.log"
    stderr_log_path = output_dir / "phase11m1_stderr.log"

    preflight_rows: list[dict[str, Any]] = []
    phase11m0_summary = read_json_if_exists(phase11m0_summary_path)
    phase11m0_summary_available = phase11m0_summary is not None
    add_check(
        preflight_rows,
        "phase11m0_summary_exists",
        phase11m0_summary_available,
        "error",
        str(phase11m0_summary_path.exists()),
        "True",
        "Phase 11M.1 requires the Phase 11M.0 summary JSON.",
    )

    phase11m0_keys_ok = phase11m0_summary is not None and REQUIRED_PHASE11M0_KEYS.issubset(phase11m0_summary)
    add_check(
        preflight_rows,
        "phase11m0_required_keys_present",
        phase11m0_keys_ok,
        "error",
        sorted(list(phase11m0_summary.keys())) if phase11m0_summary else [],
        sorted(REQUIRED_PHASE11M0_KEYS),
        "Phase 11M.0 summary must expose the fields required by the execution wrapper.",
    )

    phase11m0_status = str(phase11m0_summary.get("status", "")) if phase11m0_summary else ""
    phase11m0_passed = (
        phase11m0_summary is not None
        and phase11m0_status == "phase11m0_approved_test_evaluation_prepare_only_passed"
        and phase11m0_summary.get("phase11l_passed") is True
        and phase11m0_summary.get("baseline_checkpoint_candidate_accepted_from_phase11l") is True
        and phase11m0_summary.get("locked_eval_command_prepared") is True
        and phase11m0_summary.get("approved_for_phase11m1_execution") is False
        and phase11m0_summary.get("next_allowed_step")
        == "collect_phase11m1_explicit_evaluation_execution_approval_or_execute_with_flag"
    )
    add_check(
        preflight_rows,
        "phase11m0_passed_gate",
        phase11m0_passed,
        "error",
        {
            "status": phase11m0_status,
            "phase11l_passed": phase11m0_summary.get("phase11l_passed") if phase11m0_summary else None,
            "baseline_checkpoint_candidate_accepted_from_phase11l": (
                phase11m0_summary.get("baseline_checkpoint_candidate_accepted_from_phase11l")
                if phase11m0_summary
                else None
            ),
            "locked_eval_command_prepared": phase11m0_summary.get("locked_eval_command_prepared") if phase11m0_summary else None,
        },
        {
            "status": "phase11m0_approved_test_evaluation_prepare_only_passed",
            "phase11l_passed": True,
            "baseline_checkpoint_candidate_accepted_from_phase11l": True,
            "locked_eval_command_prepared": True,
        },
        "Phase 11M.0 must pass before execution can be wrapped.",
    )

    phase11m0_model_path = str(phase11m0_summary.get("accepted_checkpoint_path", "")) if phase11m0_summary else ""
    dataset_yaml_path_from_phase11m0 = str(phase11m0_summary.get("dataset_yaml_path", "")) if phase11m0_summary else ""
    runtime_model_path = model_path_override.strip() or phase11m0_model_path
    runtime_data_yaml = data_yaml_override.strip() or dataset_yaml_path_from_phase11m0
    runtime_project = project_override.strip() or default_project_path()
    runtime_target = detect_runtime_target()

    runtime_model = Path(runtime_model_path) if runtime_model_path else None
    runtime_model_exists = runtime_model is not None and runtime_model.exists()
    runtime_model_non_empty = runtime_model_exists and runtime_model.stat().st_size > 0
    runtime_data_yaml_path = Path(runtime_data_yaml) if runtime_data_yaml else None
    runtime_data_yaml_exists = runtime_data_yaml_path is not None and runtime_data_yaml_path.exists()
    ready_for_kaggle_execution_candidate = bool(
        phase11m0_summary.get("ready_for_kaggle_execution_candidate") if phase11m0_summary else False
    )

    add_check(
        preflight_rows,
        "runtime_model_path_resolved",
        bool(runtime_model_path),
        "error",
        runtime_model_path,
        "non-empty runtime model path",
        "Runtime model path comes from Phase 11M.0 unless overridden.",
    )
    add_check(
        preflight_rows,
        "runtime_model_exists",
        runtime_model_exists,
        "error",
        str(runtime_model_exists),
        "True",
        "Runtime model path must exist before execution.",
    )
    add_check(
        preflight_rows,
        "runtime_model_non_empty",
        runtime_model_non_empty,
        "error",
        runtime_model.stat().st_size if runtime_model_exists else 0,
        "> 0",
        "Runtime model file must be non-empty before execution.",
    )
    add_check(
        preflight_rows,
        "runtime_data_yaml_resolved",
        bool(runtime_data_yaml),
        "error",
        runtime_data_yaml,
        "non-empty runtime dataset yaml path",
        "Runtime dataset YAML comes from Phase 11M.0 unless overridden.",
    )
    add_check(
        preflight_rows,
        "runtime_data_yaml_exists",
        runtime_data_yaml_exists,
        "error",
        str(runtime_data_yaml_exists),
        "True",
        "Runtime dataset YAML must exist before execution.",
    )

    execution_ready = runtime_model_exists and runtime_model_non_empty and runtime_data_yaml_exists
    blocked_reason = ""
    if not runtime_data_yaml_exists and runtime_data_yaml.startswith("/kaggle/"):
        blocked_reason = "runtime_dataset_yaml_not_available"
    elif not runtime_model_exists:
        blocked_reason = "runtime_model_path_not_available"
    elif not runtime_model_non_empty:
        blocked_reason = "runtime_model_empty"
    elif not runtime_data_yaml_exists:
        blocked_reason = "runtime_dataset_yaml_not_available"

    approval_rows = build_approval_template_rows()
    approval_validation = validate_approval_csv(approval_csv_path)
    execution_requested = execute
    approval_csv_used = approval_validation["provided"]
    approval_passed = approval_validation["passed"]
    execution_allowed = execution_requested or approval_passed

    locked_command = build_runtime_command(
        model_path=runtime_model_path,
        data_yaml=runtime_data_yaml,
        project=runtime_project,
        run_name=run_name,
        imgsz=imgsz,
        batch=batch,
        split=split,
    )
    add_check(
        preflight_rows,
        "locked_runtime_command_prepared",
        bool(locked_command["command"]),
        "error",
        locked_command["command"],
        "non-empty locked runtime command",
        "Phase 11M.1 always prepares the exact runtime command before any allowed execution.",
    )
    add_check(
        preflight_rows,
        "ready_for_kaggle_execution_candidate_recorded",
        True,
        "info",
        ready_for_kaggle_execution_candidate,
        True,
        "Kaggle runtime remains the intended execution target when mounted paths are available.",
    )

    locked_runtime_command_executed = False
    evaluation_executed = False
    inference_executed = False
    prediction_executed = False
    subprocess_eval_command_executed = False
    returncode: int | None = None
    metrics_summary: dict[str, Any] | None = None

    stdout_log_path.write_text("", encoding="utf-8")
    stderr_log_path.write_text("", encoding="utf-8")

    if not execution_allowed:
        status = STATUS_BLOCKED_DEFAULT
        next_allowed_step = NEXT_BLOCKED_DEFAULT
    elif not execution_ready:
        status = STATUS_BLOCKED_PREFLIGHT
        next_allowed_step = NEXT_BLOCKED_PREFLIGHT
    else:
        start_time = utc_now()
        subprocess_eval_command_executed = True
        locked_runtime_command_executed = True
        with stdout_log_path.open("w", encoding="utf-8") as stdout_handle, stderr_log_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                locked_command["command_tokens"],
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        returncode = completed.returncode
        end_time = utc_now()
        evaluation_executed = True
        if returncode == 0:
            status = STATUS_EXECUTED
            next_allowed_step = NEXT_EXECUTED
        else:
            status = STATUS_FAILED
            next_allowed_step = NEXT_FAILED
        metrics_summary = collect_metrics_and_outputs(runtime_project, run_name, end_time)

    if metrics_summary is None:
        metrics_summary = collect_metrics_and_outputs(runtime_project, run_name, utc_now(), require_existing=False)
    output_manifest_rows = build_output_manifest_rows(runtime_project, run_name)

    summary = {
        "phase": PHASE,
        "status": status,
        "phase11m0_summary_available": phase11m0_summary_available,
        "phase11m0_status": phase11m0_status,
        "phase11m0_passed": phase11m0_passed,
        "phase11l_checkpoint_candidate_accepted": (
            phase11m0_summary.get("baseline_checkpoint_candidate_accepted_from_phase11l") if phase11m0_summary else False
        ),
        "phase11m0_model_path": phase11m0_model_path,
        "runtime_model_path": runtime_model_path,
        "runtime_model_exists": runtime_model_exists,
        "runtime_model_non_empty": runtime_model_non_empty,
        "checkpoint_loaded": False,
        "checkpoint_copied": False,
        "dataset_yaml_path_from_phase11m0": dataset_yaml_path_from_phase11m0,
        "runtime_data_yaml": runtime_data_yaml,
        "runtime_data_yaml_exists": runtime_data_yaml_exists,
        "runtime_target": runtime_target,
        "ready_for_kaggle_execution_candidate": ready_for_kaggle_execution_candidate,
        "execution_ready": execution_ready,
        "blocked_reason": blocked_reason,
        "execution_requested": execution_requested,
        "approval_csv_used": approval_csv_used,
        "approval_passed": approval_passed,
        "execution_allowed": execution_allowed,
        "locked_runtime_command_prepared": True,
        "locked_runtime_command_executed": locked_runtime_command_executed,
        "training_executed": False,
        "evaluation_executed": evaluation_executed,
        "inference_executed": inference_executed,
        "prediction_executed": prediction_executed,
        "export_executed": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "weights_modified": False,
        "weights_copied_to_artifacts": False,
        "ultralytics_python_api_imported": False,
        "subprocess_eval_command_executed": subprocess_eval_command_executed,
        "returncode": returncode,
        "output_dir": str(output_dir),
        "runtime_project": runtime_project,
        "runtime_name": run_name,
        "stdout_log_path": str(stdout_log_path),
        "stderr_log_path": str(stderr_log_path),
        "metrics_summary_available": metrics_summary["metrics_summary_available"],
        "best_epoch_from_phase11l": phase11m0_summary.get("best_epoch_from_phase11l") if phase11m0_summary else None,
        "best_metric_map50_95_from_phase11l": (
            phase11m0_summary.get("best_metric_map50_95_from_phase11l") if phase11m0_summary else None
        ),
        "provenance_caveat_carried_forward": (
            phase11m0_summary.get("provenance_caveat_carried_forward") if phase11m0_summary else ""
        ),
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": utc_now(),
    }

    command_json = {
        "phase": PHASE,
        "command": locked_command["command"],
        "command_tokens": locked_command["command_tokens"],
        "commented_command": locked_command["commented_command"],
        "execution_requested": execution_requested,
        "approval_csv_used": approval_csv_used,
        "approval_passed": approval_passed,
        "execution_allowed": execution_allowed,
        "executed": locked_runtime_command_executed,
    }
    manifest = build_execution_manifest(summary)

    write_json(summary, output_dir / "phase11m1_execution_summary.json")
    write_csv(preflight_rows, output_dir / "phase11m1_execution_preflight_checks.csv", PREFLIGHT_FIELDS)
    write_json(command_json, output_dir / "phase11m1_locked_runtime_command.json")
    (output_dir / "phase11m1_locked_runtime_command.sh").write_text(
        build_locked_shell_script(locked_command["commented_command"]), encoding="utf-8"
    )
    write_csv(approval_rows, output_dir / "phase11m1_evaluation_approval_template.csv", APPROVAL_FIELDS)
    write_json(manifest, output_dir / "phase11m1_non_execution_or_execution_manifest.json")
    write_csv(output_manifest_rows, output_dir / "phase11m1_output_manifest.csv", OUTPUT_MANIFEST_FIELDS)
    (output_dir / "README.md").write_text(build_readme(summary), encoding="utf-8")
    if metrics_summary["metrics_summary_available"]:
        write_json(metrics_summary, output_dir / "phase11m1_metrics_summary.json")
        write_csv(metrics_summary["metric_rows"], output_dir / "phase11m1_metrics_summary.csv", ["metric", "value"])
    return summary


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_optional_repo_path(path_str: str) -> Path | None:
    if not path_str:
        return None
    return resolve_repo_path(path_str)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def default_project_path() -> str:
    if Path("/kaggle/working").exists():
        return "/kaggle/working/phase11m_test_eval"
    return DEFAULT_REPO_PROJECT


def detect_runtime_target() -> str:
    if Path("/kaggle/working").exists() and Path("/kaggle/input").exists():
        return "kaggle_gpu_or_kaggle_like_runtime"
    return "local_or_non_kaggle_runtime"


def build_approval_template_rows() -> list[dict[str, str]]:
    return [
        {
            "phase": PHASE,
            "approval_item": item,
            "approved": "false",
            "approver": "",
            "approval_timestamp": "",
            "notes": "Explicit approval required before any evaluation execution.",
        }
        for item in APPROVAL_ITEMS
    ]


def validate_approval_csv(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "passed": False, "notes": "No approval CSV provided."}
    if not path.exists():
        return {"provided": True, "passed": False, "notes": f"Approval CSV does not exist: {path}"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    seen = {row.get("approval_item", "").strip(): row for row in rows}
    missing = [item for item in APPROVAL_ITEMS if item not in seen]
    if missing:
        return {"provided": True, "passed": False, "notes": f"Missing approval rows: {', '.join(missing)}"}
    for item in APPROVAL_ITEMS:
        approved = str(seen[item].get("approved", "")).strip().lower()
        if approved != "true":
            return {"provided": True, "passed": False, "notes": f"Approval row not explicitly true: {item}"}
    return {"provided": True, "passed": True, "notes": f"All required approval rows explicitly true in {path}"}


def build_runtime_command(
    model_path: str,
    data_yaml: str,
    project: str,
    run_name: str,
    imgsz: int,
    batch: int,
    split: str,
) -> dict[str, Any]:
    tokens = [
        "yolo",
        "detect",
        "val",
        f"model={model_path}",
        f"data={data_yaml}",
        f"split={split}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"project={project}",
        f"name={run_name}",
        "save_json=True",
        "save_conf=True",
        "plots=True",
    ]
    command = shlex.join(tokens)
    return {"command": command, "command_tokens": tokens, "commented_command": "# " + command}


def build_locked_shell_script(commented_command: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'echo "This command file is a Phase 11M.1 execution wrapper artifact."',
            'echo "Do not execute it directly unless Phase 11M.1 execution has been explicitly approved."',
            "exit 1",
            "",
            "# Locked runtime command below for review only:",
            commented_command,
            "",
        ]
    )


def collect_metrics_and_outputs(
    runtime_project: str,
    run_name: str,
    generated_at_utc: str,
    require_existing: bool = True,
) -> dict[str, Any]:
    run_dir = Path(runtime_project) / run_name
    output_files = [
        run_dir / "results.json",
        run_dir / "results.csv",
        run_dir / "args.yaml",
        run_dir / "confusion_matrix.png",
        run_dir / "confusion_matrix_normalized.png",
        run_dir / "predictions.json",
    ]
    found_paths = [str(path) for path in output_files if path.exists()]
    metrics_summary_available = False
    metric_rows: list[dict[str, Any]] = []
    parsed_from = ""
    if (run_dir / "results.json").exists():
        try:
            results_payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results_payload = {}
        if isinstance(results_payload, dict):
            for key, value in results_payload.items():
                if isinstance(value, (int, float, str, bool)):
                    metric_rows.append({"metric": key, "value": value})
            metrics_summary_available = len(metric_rows) > 0
            parsed_from = str(run_dir / "results.json")
    elif (run_dir / "results.csv").exists():
        with (run_dir / "results.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if rows:
            last_row = rows[-1]
            for key, value in last_row.items():
                if value != "":
                    metric_rows.append({"metric": key, "value": value})
            metrics_summary_available = len(metric_rows) > 0
            parsed_from = str(run_dir / "results.csv")

    if require_existing and not run_dir.exists():
        found_paths = []
        metrics_summary_available = False
        metric_rows = []
        parsed_from = ""

    return {
        "metrics_summary_available": metrics_summary_available,
        "parsed_from": parsed_from,
        "output_run_dir": str(run_dir),
        "found_output_paths": found_paths,
        "metric_rows": metric_rows,
        "generated_at_utc": generated_at_utc,
    }


def build_output_manifest_rows(runtime_project: str, run_name: str) -> list[dict[str, Any]]:
    run_dir = Path(runtime_project) / run_name
    if not run_dir.exists():
        return [
            {
                "relative_path": str(run_dir),
                "exists": False,
                "is_dir": True,
                "size_bytes": 0,
                "modified_time_utc": "",
                "note": "Evaluation output directory not present.",
            }
        ]
    rows = []
    for path in sorted(run_dir.rglob("*")):
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(path.relative_to(run_dir)),
                "exists": True,
                "is_dir": path.is_dir(),
                "size_bytes": 0 if path.is_dir() else stat.st_size,
                "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "note": "Evaluation output discovered by Phase 11M.1.",
            }
        )
    return rows


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


def build_execution_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_executed": False,
        "evaluation_executed": summary["evaluation_executed"],
        "inference_executed": summary["inference_executed"],
        "prediction_executed": summary["prediction_executed"],
        "export_executed": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "weights_modified": False,
        "weights_copied_to_artifacts": False,
        "checkpoint_loaded": False,
        "ultralytics_python_api_imported": False,
        "subprocess_eval_command_executed": summary["subprocess_eval_command_executed"],
    }


def build_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 11M.1 Approved Test Evaluation Execution",
            "",
            "Phase 11M.1 is blocked by default and executes only with explicit approval.",
            "",
            f"- status = `{summary['status']}`",
            f"- runtime_target = `{summary['runtime_target']}`",
            f"- runtime_model_path = `{summary['runtime_model_path']}`",
            f"- runtime_data_yaml = `{summary['runtime_data_yaml']}`",
            f"- execution_requested = `{summary['execution_requested']}`",
            f"- approval_csv_used = `{summary['approval_csv_used']}`",
            f"- approval_passed = `{summary['approval_passed']}`",
            f"- execution_allowed = `{summary['execution_allowed']}`",
            f"- locked_runtime_command_executed = `{summary['locked_runtime_command_executed']}`",
            f"- next_allowed_step = `{summary['next_allowed_step']}`",
            "",
            "This phase never trains, never mutates datasets or labels, never copies or modifies weights, and never loads checkpoint tensors directly in Python.",
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
