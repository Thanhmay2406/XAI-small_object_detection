"""Execute the locked Phase 11J.1 Kaggle training command only when explicitly allowed."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "11J.1"
STATUS_DRY_RUN = "phase11j1_execution_not_started_missing_execute_flag"
STATUS_NOT_KAGGLE = "phase11j1_blocked_not_kaggle_runtime"
STATUS_EXECUTED = "phase11j1_locked_kaggle_training_executed"
STATUS_FAILED = "phase11j1_locked_kaggle_training_failed"
NEXT_DRY_RUN = "rerun_phase11j1_with_execute_on_kaggle_runtime"
NEXT_EXECUTED = "phase11k_collect_training_outputs_and_metrics"
NEXT_FAILED = "inspect_phase11j1_failure_logs"
REPO_ROOT_REQUIRED_DIRS = ("artifacts", "docs", "scripts", "src")
OUTPUT_DIR_DEFAULT = "artifacts/phase11j1_locked_kaggle_training_execution"
PHASE11J0_SUMMARY_DEFAULT = (
    "artifacts/phase11j_approved_kaggle_training_command_lock/phase11j0_training_command_lock_summary.json"
)
LOCKED_COMMAND_DEFAULT = (
    "artifacts/phase11j_approved_kaggle_training_command_lock/phase11j0_approved_training_command.txt"
)
EXPECTED_BEST_PT = "/kaggle/working/phase11j_training/yolov8n_drill_bit_phase11j/weights/best.pt"
EXPECTED_LAST_PT = "/kaggle/working/phase11j_training/yolov8n_drill_bit_phase11j/weights/last.pt"
EXPECTED_RESULTS_CSV = "/kaggle/working/phase11j_training/yolov8n_drill_bit_phase11j/results.csv"
EXPECTED_RUN_DIR = "/kaggle/working/phase11j_training/yolov8n_drill_bit_phase11j"
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the locked Phase 11J.1 Kaggle training command only with --execute."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--phase11j0-summary", default=PHASE11J0_SUMMARY_DEFAULT)
    parser.add_argument("--locked-command-file", default=LOCKED_COMMAND_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    phase11j0_summary_path = (repo_root / args.phase11j0_summary).resolve()
    locked_command_path = (repo_root / args.locked_command_file).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    ensure_repo_root(repo_root)
    phase11j0_summary = read_json_required(phase11j0_summary_path, "Phase 11J.0 summary")
    validate_phase11j0_summary(phase11j0_summary)
    locked_command = read_locked_command(locked_command_path)
    validate_command_match(phase11j0_summary, locked_command)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_phase11j1(
        output_dir=output_dir,
        phase11j0_summary=phase11j0_summary,
        phase11j0_summary_path=phase11j0_summary_path,
        locked_command_path=locked_command_path,
        locked_command=locked_command,
        execute=args.execute,
    )
    write_outputs(output_dir, result)
    print(json.dumps(result["summary"], indent=2))


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


def read_locked_command(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Locked command file not found: {path}")
    command = path.read_text(encoding="utf-8").strip()
    if not command:
        raise SystemExit(f"Locked command file is empty: {path}")
    return command


def validate_phase11j0_summary(summary: dict[str, Any]) -> None:
    failures: list[str] = []
    expected = {
        "status": "phase11j0_approved_kaggle_training_command_locked",
        "training_command_locked": True,
        "training_allowed": True,
        "ready_for_training_execution": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created": False,
        "next_allowed_step": "phase11j1_execute_locked_kaggle_training",
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            failures.append(f"phase11j0_summary.{key} must equal {expected_value!r}; got {summary.get(key)!r}")
    if failures:
        raise SystemExit("Phase 11J.0 validation failed:\n- " + "\n- ".join(failures))


def validate_command_match(summary: dict[str, Any], locked_command: str) -> None:
    approved_command = str(summary.get("approved_training_command", "")).strip()
    if not approved_command:
        raise SystemExit("Phase 11J.0 summary does not contain approved_training_command")
    if locked_command != approved_command:
        raise SystemExit("Locked command file does not match approved_training_command in Phase 11J.0 summary")


def run_phase11j1(
    output_dir: Path,
    phase11j0_summary: dict[str, Any],
    phase11j0_summary_path: Path,
    locked_command_path: Path,
    locked_command: str,
    execute: bool,
) -> dict[str, Any]:
    stdout_path = output_dir / "phase11j1_execution_stdout.log"
    stderr_path = output_dir / "phase11j1_execution_stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    dataset_root = str(phase11j0_summary["dataset_root"])
    yaml_path = str(phase11j0_summary["yaml_path"])
    expected_outputs = build_expected_outputs_check(phase11j0_summary)

    if not execute:
        summary = build_summary(
            status=STATUS_DRY_RUN,
            phase11j0_summary=phase11j0_summary,
            phase11j0_summary_path=phase11j0_summary_path,
            locked_command_path=locked_command_path,
            locked_command=locked_command,
            training_executed=False,
            return_code=None,
            yolo_internal_validation=False,
            weights_created=False,
            next_allowed_step=NEXT_DRY_RUN,
            execution_started_at_utc=None,
            execution_finished_at_utc=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            failure_reason="Missing --execute flag; dry-run only.",
        )
        return {
            "summary": summary,
            "locked_command": locked_command,
            "expected_outputs": expected_outputs,
            "non_mutation_manifest": build_non_mutation_manifest(
                status=STATUS_DRY_RUN,
                training_executed=False,
                weights_created=False,
            ),
            "readme_text": build_readme(summary),
        }

    kaggle_checks = validate_kaggle_runtime(dataset_root=dataset_root, yaml_path=yaml_path)
    if not kaggle_checks["ok"]:
        stderr_path.write_text("\n".join(kaggle_checks["errors"]) + "\n", encoding="utf-8")
        summary = build_summary(
            status=STATUS_NOT_KAGGLE,
            phase11j0_summary=phase11j0_summary,
            phase11j0_summary_path=phase11j0_summary_path,
            locked_command_path=locked_command_path,
            locked_command=locked_command,
            training_executed=False,
            return_code=None,
            yolo_internal_validation=False,
            weights_created=False,
            next_allowed_step=NEXT_DRY_RUN,
            execution_started_at_utc=None,
            execution_finished_at_utc=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            failure_reason="Environment does not satisfy required Kaggle runtime checks.",
        )
        expected_outputs["runtime_environment_checks"] = kaggle_checks
        return {
            "summary": summary,
            "locked_command": locked_command,
            "expected_outputs": expected_outputs,
            "non_mutation_manifest": build_non_mutation_manifest(
                status=STATUS_NOT_KAGGLE,
                training_executed=False,
                weights_created=False,
            ),
            "readme_text": build_readme(summary),
        }

    command_tokens = shlex.split(locked_command)
    start_time = datetime.now(timezone.utc).isoformat()
    training_started = False
    return_code: int | None = None
    try:
        training_started = True
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command_tokens,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        return_code = completed.returncode
    finally:
        end_time = datetime.now(timezone.utc).isoformat()

    weights_created = Path(EXPECTED_BEST_PT).exists() or Path(EXPECTED_LAST_PT).exists()
    status = STATUS_EXECUTED if return_code == 0 else STATUS_FAILED
    next_allowed_step = NEXT_EXECUTED if return_code == 0 else NEXT_FAILED
    expected_outputs = build_expected_outputs_check(phase11j0_summary)
    expected_outputs["runtime_environment_checks"] = kaggle_checks

    summary = build_summary(
        status=status,
        phase11j0_summary=phase11j0_summary,
        phase11j0_summary_path=phase11j0_summary_path,
        locked_command_path=locked_command_path,
        locked_command=locked_command,
        training_executed=training_started,
        return_code=return_code,
        yolo_internal_validation=return_code == 0,
        weights_created=weights_created,
        next_allowed_step=next_allowed_step,
        execution_started_at_utc=start_time,
        execution_finished_at_utc=end_time,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        failure_reason="" if return_code == 0 else "Locked Kaggle training command returned a non-zero exit code.",
    )
    return {
        "summary": summary,
        "locked_command": locked_command,
        "expected_outputs": expected_outputs,
        "non_mutation_manifest": build_non_mutation_manifest(
            status=status,
            training_executed=training_started,
            weights_created=weights_created,
        ),
        "readme_text": build_readme(summary),
    }


def validate_kaggle_runtime(dataset_root: str, yaml_path: str) -> dict[str, Any]:
    checks = [
        ("kaggle_working_exists", KAGGLE_WORKING.exists(), str(KAGGLE_WORKING)),
        ("kaggle_input_exists", KAGGLE_INPUT.exists(), str(KAGGLE_INPUT)),
        ("locked_yaml_exists", Path(yaml_path).exists(), yaml_path),
        ("locked_dataset_root_exists", Path(dataset_root).exists(), dataset_root),
    ]
    errors = [f"{check_id} failed: expected existing path {path}" for check_id, ok, path in checks if not ok]
    return {
        "ok": not errors,
        "checks": [
            {"check_id": check_id, "passed": ok, "path": path}
            for check_id, ok, path in checks
        ],
        "errors": errors,
    }


def build_expected_outputs_check(phase11j0_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "expected_output_dir": phase11j0_summary["expected_output_dir"],
        "expected_best_pt": {"path": EXPECTED_BEST_PT, "exists": Path(EXPECTED_BEST_PT).exists()},
        "expected_last_pt": {"path": EXPECTED_LAST_PT, "exists": Path(EXPECTED_LAST_PT).exists()},
        "expected_results_csv": {"path": EXPECTED_RESULTS_CSV, "exists": Path(EXPECTED_RESULTS_CSV).exists()},
        "expected_run_dir": {"path": EXPECTED_RUN_DIR, "exists": Path(EXPECTED_RUN_DIR).exists()},
        "notes": "Existence values reflect the current filesystem snapshot when Phase 11J.1 ran.",
    }


def build_non_mutation_manifest(status: str, training_executed: bool, weights_created: bool) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": status,
        "training_executed": training_executed,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created": weights_created,
        "checkpoints_created": weights_created,
        "notes": "Phase 11J.1 only executes the locked Kaggle training command when explicitly requested. It never mutates datasets or uploads anything.",
    }


def build_summary(
    *,
    status: str,
    phase11j0_summary: dict[str, Any],
    phase11j0_summary_path: Path,
    locked_command_path: Path,
    locked_command: str,
    training_executed: bool,
    return_code: int | None,
    yolo_internal_validation: bool,
    weights_created: bool,
    next_allowed_step: str,
    execution_started_at_utc: str | None,
    execution_finished_at_utc: str | None,
    stdout_path: Path,
    stderr_path: Path,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": status,
        "phase11j0_status": phase11j0_summary["status"],
        "phase11j0_summary_path": str(phase11j0_summary_path),
        "locked_command_file": str(locked_command_path),
        "locked_command_matches_summary": locked_command == phase11j0_summary["approved_training_command"],
        "training_command_locked": True,
        "approved_training_command": locked_command,
        "dataset_root": phase11j0_summary["dataset_root"],
        "yaml_path": phase11j0_summary["yaml_path"],
        "expected_output_dir": phase11j0_summary["expected_output_dir"],
        "training_allowed": True,
        "ready_for_training_execution": True,
        "training_executed": training_executed,
        "evaluation_executed": False,
        "yolo_internal_validation": yolo_internal_validation,
        "inference_executed": False,
        "dataset_mutated": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created": weights_created,
        "weights_or_checkpoints_created": weights_created,
        "expected_best_pt": EXPECTED_BEST_PT,
        "expected_last_pt": EXPECTED_LAST_PT,
        "expected_results_csv": EXPECTED_RESULTS_CSV,
        "return_code": return_code,
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
        "execution_started_at_utc": execution_started_at_utc,
        "execution_finished_at_utc": execution_finished_at_utc,
        "failure_reason": failure_reason,
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    write_json(output_dir / "phase11j1_training_execution_summary.json", result["summary"])
    (output_dir / "phase11j1_locked_command_used.txt").write_text(
        result["locked_command"] + "\n", encoding="utf-8"
    )
    write_json(output_dir / "phase11j1_expected_outputs_check.json", result["expected_outputs"])
    write_json(output_dir / "phase11j1_non_mutation_manifest.json", result["non_mutation_manifest"])
    (output_dir / "README.md").write_text(result["readme_text"], encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 11J.1 Locked Kaggle Training Execution",
            "",
            "Phase 11J.1 executes the locked Kaggle training command only when `--execute` is supplied in a valid Kaggle runtime.",
            "",
            "## Outcome",
            f"- status = `{summary['status']}`",
            f"- training_executed = `{summary['training_executed']}`",
            f"- evaluation_executed = `{summary['evaluation_executed']}`",
            f"- yolo_internal_validation = `{summary['yolo_internal_validation']}`",
            f"- inference_executed = `{summary['inference_executed']}`",
            f"- weights_created = `{summary['weights_created']}`",
            f"- return_code = `{summary['return_code']}`",
            f"- next_allowed_step = `{summary['next_allowed_step']}`",
            "",
            "## Guardrails",
            "- The locked command is used exactly as approved.",
            "- No dataset mutation is performed.",
            "- No Kaggle upload is performed.",
            "- No separate evaluation or inference command is run.",
        ]
    ) + "\n"


if __name__ == "__main__":
    main()
