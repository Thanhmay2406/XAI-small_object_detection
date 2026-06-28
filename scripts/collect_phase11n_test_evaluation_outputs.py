"""Collect and validate Phase 11N test evaluation outputs without execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11N"
DEFAULT_EVAL_DIR = "experiments/phase11m_test_eval/yolov8n_drill_bit_phase11m_test_eval"
DEFAULT_TRAINING_DIR = "experiments/phase11j_training"
DEFAULT_OUTPUT_DIR = "artifacts/phase11n_test_evaluation_output_collection_and_validation"
PHASE11L_SUMMARY_DEFAULT = (
    "artifacts/phase11l_training_output_integrity_and_provenance/phase11l_training_output_integrity_summary.json"
)
PHASE11M1_SUMMARY_DEFAULT = (
    "artifacts/phase11m1_approved_test_evaluation_execution/phase11m1_execution_summary.json"
)
PHASE11M1_STDOUT_DEFAULT = (
    "artifacts/phase11m1_approved_test_evaluation_execution/phase11m1_stdout.log"
)
STATUS_PASS = "phase11n_test_evaluation_outputs_collected_and_validated"
STATUS_NEEDS_REVIEW = "phase11n_test_evaluation_outputs_collected_needs_manual_metric_review"
STATUS_FAIL = "phase11n_test_evaluation_outputs_missing_or_invalid"
NEXT_PASS = "phase11o_test_error_case_selection_and_xai_evidence_planning"
NEXT_REVIEW = "manually_extract_phase11m1_metrics_then_rerun_phase11n_or_continue_with_caveat"
NEXT_FAIL = "inspect_phase11m1_execution_output_path_before_phase11n"
HASH_SKIP_SIZE_BYTES = 20_000_000
HASH_CHUNK_SIZE = 1024 * 1024
MANIFEST_FIELDS = [
    "relative_path",
    "absolute_path",
    "suffix",
    "size_bytes",
    "mtime_utc",
    "sha256",
    "sha256_skipped_large_file",
]
METRIC_FIELDS = ["metric", "value", "source", "notes"]
VALIDATION_FIELDS = ["check_name", "passed", "severity", "observed_value", "expected_value", "notes"]
EXPECTED_ARTIFACT_FIELDS = [
    "artifact_name",
    "expected",
    "present",
    "path",
    "size_bytes",
    "required_for_phase11n_pass",
    "notes",
]
EXPECTED_FILES = [
    "args.yaml",
    "predictions.json",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "PR_curve.png",
    "results.csv",
    "results.json",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and validate Phase 11M.1 test evaluation outputs without executing anything."
    )
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--training-dir", default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phase11l-summary", default=PHASE11L_SUMMARY_DEFAULT)
    parser.add_argument("--phase11m1-summary", default=PHASE11M1_SUMMARY_DEFAULT)
    parser.add_argument("--phase11m1-stdout-log", default=PHASE11M1_STDOUT_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = collect_phase11n_test_evaluation_outputs(
        eval_dir=resolve_repo_path(args.eval_dir),
        training_dir=resolve_repo_path(args.training_dir),
        output_dir=resolve_repo_path(args.output_dir),
        phase11l_summary_path=resolve_repo_path(args.phase11l_summary),
        phase11m1_summary_path=resolve_repo_path(args.phase11m1_summary),
        phase11m1_stdout_log_path=resolve_repo_path(args.phase11m1_stdout_log),
    )
    concise = {
        "phase": summary["phase"],
        "status": summary["status"],
        "eval_dir": summary["eval_dir"],
        "metrics_summary_available": summary["metrics_summary_available"],
        "evaluation_outputs_validated": summary["evaluation_outputs_validated"],
        "needs_manual_metric_review": summary["needs_manual_metric_review"],
        "next_allowed_step": summary["next_allowed_step"],
    }
    print(json.dumps(concise, indent=2))


def collect_phase11n_test_evaluation_outputs(
    eval_dir: Path,
    training_dir: Path,
    output_dir: Path,
    phase11l_summary_path: Path,
    phase11m1_summary_path: Path,
    phase11m1_stdout_log_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    phase11l_summary = read_json_if_exists(phase11l_summary_path)
    phase11m1_summary = read_json_if_exists(phase11m1_summary_path)
    eval_dir_exists = eval_dir.exists() and eval_dir.is_dir()
    training_dir_exists = training_dir.exists() and training_dir.is_dir()
    manifest_rows = build_manifest_rows(eval_dir)
    eval_dir_file_count = len(manifest_rows)
    eval_dir_total_size_bytes = sum(int(row["size_bytes"]) for row in manifest_rows)

    validation_rows: list[dict[str, Any]] = []
    add_check(validation_rows, "eval_dir_exists", eval_dir_exists, "error", str(eval_dir_exists), "True", "Evaluation output directory must exist.")
    add_check(
        validation_rows,
        "eval_dir_non_empty",
        eval_dir_file_count > 0,
        "error",
        eval_dir_file_count,
        "> 0",
        "Evaluation output directory must contain files.",
    )
    add_check(
        validation_rows,
        "training_dir_exists",
        training_dir_exists,
        "info",
        str(training_dir_exists),
        "True",
        "Training output directory is optional but should exist for provenance linkage.",
    )

    path_map = {Path(row["relative_path"]).name: row for row in manifest_rows}
    presence_rows = build_expected_artifact_rows(eval_dir, path_map)
    args_yaml_path = eval_dir / "args.yaml"
    predictions_path = eval_dir / "predictions.json"
    confusion_matrix_path = eval_dir / "confusion_matrix.png"
    confusion_matrix_norm_path = eval_dir / "confusion_matrix_normalized.png"
    pr_curve_path = eval_dir / "BoxPR_curve.png"
    f1_curve_path = eval_dir / "BoxF1_curve.png"
    p_curve_path = eval_dir / "BoxP_curve.png"
    r_curve_path = eval_dir / "BoxR_curve.png"
    results_csv_path = eval_dir / "results.csv"
    results_json_path = eval_dir / "results.json"
    val_batch_images = sorted(eval_dir.glob("val_batch*.jpg")) if eval_dir_exists else []

    args_yaml_available = args_yaml_path.exists()
    args_yaml_text = args_yaml_path.read_text(encoding="utf-8") if args_yaml_available and args_yaml_path.stat().st_size <= 100_000 else ""
    args_yaml_mentions_split_test = "split: test" in args_yaml_text.lower() or "split=test" in args_yaml_text.lower()

    metrics_payload, parse_notes = parse_metrics(results_json_path, results_csv_path, phase11m1_stdout_log_path)
    metrics_summary_available = metrics_payload["metrics_summary_available"]
    metrics_parse_source = metrics_payload["metrics_parse_source"]
    test_map50 = metrics_payload["test_map50"]
    test_map50_95 = metrics_payload["test_map50_95"]
    test_precision = metrics_payload["test_precision"]
    test_recall = metrics_payload["test_recall"]

    predictions_json_available = predictions_path.exists()
    predictions_json_size_bytes = predictions_path.stat().st_size if predictions_json_available else 0
    confusion_matrix_available = confusion_matrix_path.exists()
    confusion_matrix_normalized_available = confusion_matrix_norm_path.exists()
    pr_curve_available = pr_curve_path.exists()
    f1_curve_available = f1_curve_path.exists()
    p_curve_available = p_curve_path.exists()
    r_curve_available = r_curve_path.exists()
    validation_batch_images_count = len(val_batch_images)

    evaluation_outputs_credible = eval_dir_exists and (
        predictions_json_available
        or confusion_matrix_available
        or confusion_matrix_normalized_available
        or validation_batch_images_count > 0
        or any([pr_curve_available, f1_curve_available, p_curve_available, r_curve_available])
    )
    needs_manual_metric_review = evaluation_outputs_credible and not metrics_summary_available
    evaluation_outputs_validated = evaluation_outputs_credible and metrics_summary_available

    add_check(
        validation_rows,
        "credible_evaluation_artifacts_found",
        evaluation_outputs_credible,
        "error",
        {
            "predictions_json_available": predictions_json_available,
            "confusion_matrix_available": confusion_matrix_available,
            "confusion_matrix_normalized_available": confusion_matrix_normalized_available,
            "validation_batch_images_count": validation_batch_images_count,
            "curve_count": sum([pr_curve_available, f1_curve_available, p_curve_available, r_curve_available]),
        },
        "at least one strong evaluation artifact family present",
        "Phase 11N requires credible output artifacts even if direct metrics are absent.",
    )
    add_check(
        validation_rows,
        "metrics_summary_available",
        metrics_summary_available,
        "info",
        metrics_parse_source,
        "results.json, results.csv, or stdout metrics",
        "Metrics may legitimately require manual review if Ultralytics did not emit machine-readable summaries.",
    )
    add_check(
        validation_rows,
        "args_yaml_mentions_split_test",
        args_yaml_mentions_split_test,
        "info",
        args_yaml_mentions_split_test,
        True,
        "This is informative only because args.yaml may be absent from validation output directories.",
    )

    if eval_dir_exists and eval_dir_file_count > 0 and metrics_summary_available:
        status = STATUS_PASS
        next_allowed_step = NEXT_PASS
    elif evaluation_outputs_credible:
        status = STATUS_NEEDS_REVIEW
        next_allowed_step = NEXT_REVIEW
    else:
        status = STATUS_FAIL
        next_allowed_step = NEXT_FAIL

    summary = {
        "phase": PHASE,
        "status": status,
        "eval_dir": str(eval_dir),
        "eval_dir_exists": eval_dir_exists,
        "eval_dir_file_count": eval_dir_file_count,
        "eval_dir_total_size_bytes": eval_dir_total_size_bytes,
        "training_dir": str(training_dir),
        "training_dir_exists": training_dir_exists,
        "phase11l_best_epoch": phase11l_summary.get("best_epoch") if phase11l_summary else None,
        "phase11l_best_metric_map50_95": phase11l_summary.get("best_metric_map50_95") if phase11l_summary else None,
        "phase11l_final_epoch": phase11l_summary.get("final_epoch") if phase11l_summary else None,
        "phase11l_final_metric_map50_95": phase11l_summary.get("final_metric_map50_95") if phase11l_summary else None,
        "phase11l_final_metric_map50": phase11l_summary.get("final_metric_map50") if phase11l_summary else None,
        "metrics_summary_available": metrics_summary_available,
        "metrics_parse_source": metrics_parse_source,
        "test_map50": test_map50,
        "test_map50_95": test_map50_95,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "predictions_json_available": predictions_json_available,
        "predictions_json_path": str(predictions_path) if predictions_json_available else "",
        "predictions_json_size_bytes": predictions_json_size_bytes,
        "confusion_matrix_available": confusion_matrix_available,
        "confusion_matrix_normalized_available": confusion_matrix_normalized_available,
        "pr_curve_available": pr_curve_available,
        "f1_curve_available": f1_curve_available,
        "p_curve_available": p_curve_available,
        "r_curve_available": r_curve_available,
        "validation_batch_images_count": validation_batch_images_count,
        "args_yaml_available": args_yaml_available,
        "args_yaml_path": str(args_yaml_path) if args_yaml_available else "",
        "args_yaml_mentions_split_test": args_yaml_mentions_split_test,
        "evaluation_outputs_credible": evaluation_outputs_credible,
        "evaluation_outputs_validated": evaluation_outputs_validated,
        "needs_manual_metric_review": needs_manual_metric_review,
        "training_executed": False,
        "evaluation_executed_by_phase11n": False,
        "inference_executed_by_phase11n": False,
        "prediction_executed_by_phase11n": False,
        "export_executed_by_phase11n": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "weights_modified": False,
        "weights_copied_to_artifacts": False,
        "checkpoint_loaded": False,
        "large_outputs_copied_to_artifacts": False,
        "provenance_caveat_carried_forward": derive_provenance_caveat(phase11l_summary, phase11m1_summary),
        "next_allowed_step": next_allowed_step,
        "generated_at_utc": utc_now(),
    }

    metrics_summary_json = {
        "phase": PHASE,
        "metrics_summary_available": metrics_summary_available,
        "metrics_parse_source": metrics_parse_source,
        "test_map50": test_map50,
        "test_map50_95": test_map50_95,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "notes": parse_notes,
        "generated_at_utc": summary["generated_at_utc"],
    }
    metric_rows = [
        {"metric": "test_map50", "value": stringify_metric(test_map50), "source": metrics_parse_source, "notes": ""},
        {"metric": "test_map50_95", "value": stringify_metric(test_map50_95), "source": metrics_parse_source, "notes": ""},
        {"metric": "test_precision", "value": stringify_metric(test_precision), "source": metrics_parse_source, "notes": ""},
        {"metric": "test_recall", "value": stringify_metric(test_recall), "source": metrics_parse_source, "notes": ""},
    ]

    write_json(summary, output_dir / "phase11n_test_evaluation_output_summary.json")
    write_csv(manifest_rows, output_dir / "phase11n_evaluation_output_manifest.csv", MANIFEST_FIELDS)
    write_json(metrics_summary_json, output_dir / "phase11n_metrics_summary.json")
    write_csv(metric_rows, output_dir / "phase11n_metrics_summary.csv", METRIC_FIELDS)
    write_csv(validation_rows, output_dir / "phase11n_validation_checks.csv", VALIDATION_FIELDS)
    write_csv(presence_rows, output_dir / "phase11n_expected_artifacts_presence.csv", EXPECTED_ARTIFACT_FIELDS)
    write_json(build_non_execution_manifest(), output_dir / "phase11n_non_execution_manifest.json")
    (output_dir / "README.md").write_text(build_readme(summary), encoding="utf-8")
    if args_yaml_available and args_yaml_text:
        (output_dir / "phase11n_args_yaml_snapshot.txt").write_text(args_yaml_text, encoding="utf-8")
    (output_dir / "phase11n_metric_parse_notes.txt").write_text("\n".join(parse_notes) + "\n", encoding="utf-8")
    return summary


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest_rows(eval_dir: Path) -> list[dict[str, Any]]:
    if not eval_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        size_bytes = stat.st_size
        skip_hash = size_bytes > HASH_SKIP_SIZE_BYTES
        rows.append(
            {
                "relative_path": str(path.relative_to(eval_dir)),
                "absolute_path": str(path),
                "suffix": path.suffix.lower(),
                "size_bytes": size_bytes,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": "" if skip_hash else sha256_file(path),
                "sha256_skipped_large_file": skip_hash,
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_expected_artifact_rows(eval_dir: Path, path_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    curve_name_map = {
        "F1_curve.png": "BoxF1_curve.png",
        "P_curve.png": "BoxP_curve.png",
        "R_curve.png": "BoxR_curve.png",
        "PR_curve.png": "BoxPR_curve.png",
    }
    for artifact_name in EXPECTED_FILES:
        actual_name = curve_name_map.get(artifact_name, artifact_name)
        record = path_map.get(actual_name)
        required = artifact_name == "predictions.json"
        rows.append(
            {
                "artifact_name": artifact_name,
                "expected": True,
                "present": record is not None,
                "path": str(eval_dir / actual_name) if record else "",
                "size_bytes": int(record["size_bytes"]) if record else 0,
                "required_for_phase11n_pass": required,
                "notes": build_expected_artifact_note(artifact_name),
            }
        )
    val_batch_images = sorted(eval_dir.glob("val_batch*.jpg")) if eval_dir.exists() else []
    rows.append(
        {
            "artifact_name": "val_batch_images",
            "expected": True,
            "present": len(val_batch_images) > 0,
            "path": str(eval_dir) if val_batch_images else "",
            "size_bytes": sum(path.stat().st_size for path in val_batch_images),
            "required_for_phase11n_pass": True,
            "notes": "Validation batch preview images are a strong sign that Ultralytics validation completed.",
        }
    )
    return rows


def build_expected_artifact_note(artifact_name: str) -> str:
    notes = {
        "args.yaml": "Optional. Validation output directories may omit args.yaml.",
        "predictions.json": "Strong evaluation artifact; records prediction outputs but not aggregate metrics by itself.",
        "confusion_matrix.png": "Strong evaluation artifact for class-level validation behavior.",
        "confusion_matrix_normalized.png": "Optional normalized confusion matrix.",
        "F1_curve.png": "Optional evaluation plot.",
        "P_curve.png": "Optional evaluation plot.",
        "R_curve.png": "Optional evaluation plot.",
        "PR_curve.png": "Optional evaluation plot.",
        "results.csv": "Useful for metrics, but not always emitted for pure validation runs.",
        "results.json": "Useful for metrics if emitted by the runtime.",
    }
    return notes.get(artifact_name, "")


def parse_metrics(results_json_path: Path, results_csv_path: Path, stdout_log_path: Path) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    payload = {
        "metrics_summary_available": False,
        "metrics_parse_source": "",
        "test_map50": None,
        "test_map50_95": None,
        "test_precision": None,
        "test_recall": None,
    }

    if results_json_path.exists():
        notes.append(f"Trying results.json: {results_json_path}")
        try:
            data = json.loads(results_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
            notes.append("results.json exists but is not valid JSON.")
        extracted = extract_metrics_from_json(data)
        if has_any_metric(extracted):
            payload.update(extracted)
            payload["metrics_summary_available"] = True
            payload["metrics_parse_source"] = str(results_json_path)
            notes.append("Parsed metrics from results.json.")
            return payload, notes
        notes.append("results.json did not expose recognized aggregate metrics.")

    if results_csv_path.exists():
        notes.append(f"Trying results.csv: {results_csv_path}")
        try:
            with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
        except OSError as exc:
            rows = []
            notes.append(f"Failed to read results.csv: {exc}")
        extracted = extract_metrics_from_csv_rows(rows)
        if has_any_metric(extracted):
            payload.update(extracted)
            payload["metrics_summary_available"] = True
            payload["metrics_parse_source"] = str(results_csv_path)
            notes.append("Parsed metrics from results.csv.")
            return payload, notes
        notes.append("results.csv did not expose recognized aggregate metrics.")

    if stdout_log_path.exists() and stdout_log_path.stat().st_size > 0:
        notes.append(f"Trying Phase 11M.1 stdout log: {stdout_log_path}")
        stdout_text = stdout_log_path.read_text(encoding="utf-8")
        extracted = extract_metrics_from_stdout(stdout_text)
        if has_any_metric(extracted):
            payload.update(extracted)
            payload["metrics_summary_available"] = True
            payload["metrics_parse_source"] = str(stdout_log_path)
            notes.append("Parsed metrics from Phase 11M.1 stdout log.")
            return payload, notes
        notes.append("Phase 11M.1 stdout log did not expose recognized aggregate metrics.")
    else:
        notes.append("Phase 11M.1 stdout log is missing or empty.")

    notes.append("No machine-readable aggregate test metrics were parsed automatically.")
    return payload, notes


def extract_metrics_from_json(data: Any) -> dict[str, float | None]:
    result = {"test_map50": None, "test_map50_95": None, "test_precision": None, "test_recall": None}
    if isinstance(data, dict):
        flattened = flatten_json(data)
        for key, value in flattened.items():
            key_lower = key.lower()
            parsed = coerce_float(value)
            if parsed is None:
                continue
            if result["test_map50_95"] is None and ("map50-95" in key_lower or "map_50_95" in key_lower):
                result["test_map50_95"] = parsed
            elif result["test_map50"] is None and "map50" in key_lower and "95" not in key_lower:
                result["test_map50"] = parsed
            elif result["test_precision"] is None and "precision" in key_lower:
                result["test_precision"] = parsed
            elif result["test_recall"] is None and "recall" in key_lower:
                result["test_recall"] = parsed
    return result


def flatten_json(data: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_json(value, new_prefix))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            flat.update(flatten_json(value, f"{prefix}[{index}]"))
    else:
        flat[prefix] = data
    return flat


def extract_metrics_from_csv_rows(rows: list[dict[str, str]]) -> dict[str, float | None]:
    result = {"test_map50": None, "test_map50_95": None, "test_precision": None, "test_recall": None}
    if not rows:
        return result
    row = rows[-1]
    for key, value in row.items():
        key_lower = key.strip().lower()
        parsed = coerce_float(value)
        if parsed is None:
            continue
        if result["test_map50_95"] is None and "map50-95" in key_lower:
            result["test_map50_95"] = parsed
        elif result["test_map50"] is None and "map50" in key_lower and "95" not in key_lower:
            result["test_map50"] = parsed
        elif result["test_precision"] is None and "precision" in key_lower:
            result["test_precision"] = parsed
        elif result["test_recall"] is None and "recall" in key_lower:
            result["test_recall"] = parsed
    return result


def extract_metrics_from_stdout(text: str) -> dict[str, float | None]:
    result = {"test_map50": None, "test_map50_95": None, "test_precision": None, "test_recall": None}
    patterns = {
        "test_map50_95": re.compile(r"mAP50-95[^0-9]*([0-9]*\\.?[0-9]+)", re.IGNORECASE),
        "test_map50": re.compile(r"mAP50(?!-95)[^0-9]*([0-9]*\\.?[0-9]+)", re.IGNORECASE),
        "test_precision": re.compile(r"precision[^0-9]*([0-9]*\\.?[0-9]+)", re.IGNORECASE),
        "test_recall": re.compile(r"recall[^0-9]*([0-9]*\\.?[0-9]+)", re.IGNORECASE),
    }
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            result[key] = coerce_float(match.group(1))
    return result


def has_any_metric(extracted: dict[str, float | None]) -> bool:
    return any(value is not None for value in extracted.values())


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def stringify_metric(value: Any) -> str:
    return "" if value is None else str(value)


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


def derive_provenance_caveat(phase11l_summary: dict[str, Any] | None, phase11m1_summary: dict[str, Any] | None) -> str:
    if phase11m1_summary and phase11m1_summary.get("provenance_caveat_carried_forward"):
        return str(phase11m1_summary["provenance_caveat_carried_forward"])
    if phase11l_summary and phase11l_summary.get("provenance_caveat"):
        return str(phase11l_summary["provenance_caveat"])
    return ""


def build_non_execution_manifest() -> dict[str, Any]:
    return {
        "training_executed": False,
        "evaluation_executed_by_phase11n": False,
        "inference_executed_by_phase11n": False,
        "prediction_executed_by_phase11n": False,
        "export_executed_by_phase11n": False,
        "dataset_mutated": False,
        "labels_mutated": False,
        "weights_modified": False,
        "weights_copied_to_artifacts": False,
        "checkpoint_loaded": False,
        "ultralytics_imported": False,
        "subprocess_eval_command_executed": False,
        "large_outputs_copied_to_artifacts": False,
    }


def build_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 11N Test Evaluation Output Collection And Validation",
            "",
            "Phase 11N is a strict non-execution collector and validator for existing Phase 11M.1 evaluation outputs.",
            "",
            f"- status = `{summary['status']}`",
            f"- eval_dir = `{summary['eval_dir']}`",
            f"- training_dir = `{summary['training_dir']}`",
            f"- metrics_summary_available = `{summary['metrics_summary_available']}`",
            f"- metrics_parse_source = `{summary['metrics_parse_source']}`",
            f"- evaluation_outputs_credible = `{summary['evaluation_outputs_credible']}`",
            f"- evaluation_outputs_validated = `{summary['evaluation_outputs_validated']}`",
            f"- needs_manual_metric_review = `{summary['needs_manual_metric_review']}`",
            f"- next_allowed_step = `{summary['next_allowed_step']}`",
            "",
            "No evaluation, inference, prediction, training, export, dataset mutation, checkpoint loading, or large-output copying was performed by Phase 11N.",
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
