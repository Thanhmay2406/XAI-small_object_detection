"""Collect Phase 11K training output provenance and metrics without executing training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE = "11K"
PHASE11J1_SUCCESS_STATUS = "phase11j1_locked_kaggle_training_executed"
PHASE11J1_SUCCESS_NEXT = "phase11k_collect_training_outputs_and_metrics"
STATUS_COMPLETE = "phase11k_training_outputs_and_metrics_collected"
STATUS_INCOMPLETE = "phase11k_training_outputs_incomplete"
DEFAULT_PHASE11J1_SUMMARY = (
    "artifacts/phase11j1_locked_kaggle_training_execution/phase11j1_training_execution_summary.json"
)
DEFAULT_TRAINING_OUTPUT_DIR = "phase11j_training/yolov8n_drill_bit_phase11j"
DEFAULT_OUTPUT_DIR = "artifacts/phase11k_training_outputs_and_metrics"
RESULTS_SUMMARY_FIELDS = [
    "phase",
    "status",
    "training_output_dir",
    "results_csv_path",
    "results_row_count",
    "metric_columns_detected",
    "best_epoch_selection_metric",
    "best_epoch_index",
    "best_epoch_value",
    "final_epoch_index",
    "final_epoch_value",
    "final_metrics_json",
    "best_metrics_json",
]
WEIGHT_MANIFEST_FIELDS = [
    "weight_label",
    "path",
    "exists",
    "size_bytes",
    "size_mb",
    "sha256",
    "modified_time_utc",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
TEXTUAL_CONFIG_CANDIDATES = ("args.yaml", "opt.yaml", "hyp.yaml")
TREE_MAX_LINES = 400
HASH_CHUNK_SIZE = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Phase 11K training outputs and metrics without running training, evaluation, or inference."
    )
    parser.add_argument("--phase11j1-summary", default=DEFAULT_PHASE11J1_SUMMARY)
    parser.add_argument("--training-output-dir", default=DEFAULT_TRAINING_OUTPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = collect_phase11k_training_outputs_and_metrics(
        phase11j1_summary_path=resolve_repo_path(args.phase11j1_summary),
        training_output_dir=resolve_repo_path(args.training_output_dir),
        output_dir=resolve_repo_path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


def collect_phase11k_training_outputs_and_metrics(
    phase11j1_summary_path: Path,
    training_output_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "phase11k_training_outputs_summary.json"
    metrics_csv_path = output_dir / "phase11k_results_metrics_summary.csv"
    weights_csv_path = output_dir / "phase11k_weight_files_manifest.csv"
    tree_path = output_dir / "phase11k_training_output_tree.txt"
    non_execution_path = output_dir / "phase11k_non_execution_manifest.json"
    readme_path = output_dir / "README.md"

    phase11j1_validation = inspect_phase11j1_summary(phase11j1_summary_path)
    training_output_inspection = inspect_training_output_dir(training_output_dir)
    metrics_info = parse_results_csv(training_output_inspection["results_csv_path"])
    weight_rows = [
        build_weight_manifest_row("best.pt", training_output_inspection["best_weight_path"]),
        build_weight_manifest_row("last.pt", training_output_inspection["last_weight_path"]),
    ]
    tree_info = build_output_tree(training_output_dir)

    results_present = training_output_inspection["results_csv_exists"]
    best_present = training_output_inspection["best_weight_exists"]
    last_present = training_output_inspection["last_weight_exists"]
    required_files_present = results_present and best_present and last_present
    metrics_parsed = metrics_info["parsed"]
    weights_hashed = all(row["sha256"] for row in weight_rows if row["exists"]) and best_present and last_present
    status = STATUS_COMPLETE if required_files_present and metrics_parsed else STATUS_INCOMPLETE
    next_allowed_step = (
        "phase11l_evaluate_trained_model_on_approved_test_split"
        if status == STATUS_COMPLETE
        else "complete_phase11k_training_output_collection_before_phase11l"
    )

    summary = {
        "phase": PHASE,
        "status": status,
        "phase11j1_summary_available": phase11j1_validation["summary_available"],
        "phase11j1_status": phase11j1_validation["summary_status"],
        "phase11j1_summary_status": phase11j1_validation["summary_status"],
        "phase11j1_summary_path": str(phase11j1_summary_path),
        "phase11j1_summary_used_for_validation": phase11j1_validation["summary_used_for_validation"],
        "phase11j1_validation_notes": phase11j1_validation["notes"],
        "training_outputs_present": training_output_inspection["training_outputs_present"],
        "training_output_dir": str(training_output_dir),
        "training_output_dir_exists": training_output_inspection["training_output_dir_exists"],
        "results_csv_path": str(training_output_inspection["results_csv_path"]),
        "best_weight_path": str(training_output_inspection["best_weight_path"]),
        "last_weight_path": str(training_output_inspection["last_weight_path"]),
        "results_csv_exists": results_present,
        "best_weight_exists": best_present,
        "last_weight_exists": last_present,
        "args_yaml_path": path_or_empty(training_output_inspection["args_yaml_path"]),
        "args_yaml_exists": training_output_inspection["args_yaml_exists"],
        "config_files_recorded": [str(path) for path in training_output_inspection["config_files_recorded"]],
        "listed_plot_and_image_files": [str(path) for path in training_output_inspection["listed_plot_and_image_files"]],
        "results_row_count": metrics_info["row_count"],
        "results_csv_parsed": metrics_info["parsed"],
        "results_csv_parse_note": metrics_info["parse_note"],
        "metric_columns_detected": metrics_info["metric_columns_detected"],
        "best_epoch_selection_metric": metrics_info["best_epoch_selection_metric"],
        "best_epoch_index": metrics_info["best_epoch_index"],
        "best_epoch_value": metrics_info["best_epoch_value"],
        "final_epoch_index": metrics_info["final_epoch_index"],
        "final_epoch_value": metrics_info["final_epoch_value"],
        "final_metrics": metrics_info["final_metrics"],
        "best_metrics": metrics_info["best_metrics"],
        "weights_hashed": weights_hashed,
        "training_output_tree_path": str(tree_path),
        "training_output_tree_truncated": tree_info["truncated"],
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created_by_phase11k": False,
        "weights_copied_by_phase11k": False,
        "output_artifacts_created": {
            "summary_json": str(summary_json_path),
            "metrics_csv": str(metrics_csv_path),
            "weights_csv": str(weights_csv_path),
            "tree_txt": str(tree_path),
            "non_execution_manifest_json": str(non_execution_path),
            "readme_md": str(readme_path),
        },
        "generated_at_utc": utc_now(),
        "next_allowed_step": next_allowed_step,
    }

    metrics_summary_row = build_metrics_summary_row(summary)
    non_execution_manifest = build_non_execution_manifest(summary)
    readme_text = build_readme(summary, weight_rows, tree_info)

    write_json(summary, summary_json_path)
    write_csv([metrics_summary_row], metrics_csv_path, RESULTS_SUMMARY_FIELDS)
    write_csv(weight_rows, weights_csv_path, WEIGHT_MANIFEST_FIELDS)
    tree_path.write_text(tree_info["text"], encoding="utf-8")
    write_json(non_execution_manifest, non_execution_path)
    readme_path.write_text(readme_text, encoding="utf-8")
    return summary


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def inspect_phase11j1_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "summary_available": False,
            "summary_status": "missing",
            "summary_used_for_validation": False,
            "notes": ["Phase 11J.1 summary file is missing; using direct local training output inspection."],
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "summary_available": False,
            "summary_status": f"invalid_json:{exc}",
            "summary_used_for_validation": False,
            "notes": ["Phase 11J.1 summary exists but is invalid JSON; using direct local training output inspection."],
        }

    status = str(payload.get("status", "missing_status"))
    summary_success = (
        status == PHASE11J1_SUCCESS_STATUS
        and payload.get("training_executed") is True
        and payload.get("weights_created") is True
        and payload.get("next_allowed_step") == PHASE11J1_SUCCESS_NEXT
    )
    if summary_success:
        return {
            "summary_available": True,
            "summary_status": status,
            "summary_used_for_validation": True,
            "notes": ["Phase 11J.1 summary passed the executed-training validation checks."],
        }

    notes = [
        "Phase 11J.1 summary exists but does not show the required executed-training success state.",
        "Phase 11K is therefore relying on direct local training output inspection for provenance collection.",
    ]
    return {
        "summary_available": False,
        "summary_status": status,
        "summary_used_for_validation": False,
        "notes": notes,
    }


def inspect_training_output_dir(training_output_dir: Path) -> dict[str, Any]:
    results_csv_path = training_output_dir / "results.csv"
    best_weight_path = training_output_dir / "weights" / "best.pt"
    last_weight_path = training_output_dir / "weights" / "last.pt"
    config_files_recorded: list[Path] = []
    args_yaml_path: Path | None = None
    for candidate_name in TEXTUAL_CONFIG_CANDIDATES:
        candidate = training_output_dir / candidate_name
        if candidate.exists():
            config_files_recorded.append(candidate)
            if candidate_name == "args.yaml":
                args_yaml_path = candidate

    listed_plot_and_image_files: list[Path] = []
    if training_output_dir.exists():
        for path in sorted(training_output_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                listed_plot_and_image_files.append(path)

    training_outputs_present = any([results_csv_path.exists(), best_weight_path.exists(), last_weight_path.exists()])
    return {
        "training_output_dir_exists": training_output_dir.exists(),
        "training_outputs_present": training_outputs_present,
        "results_csv_path": results_csv_path,
        "best_weight_path": best_weight_path,
        "last_weight_path": last_weight_path,
        "results_csv_exists": results_csv_path.exists(),
        "best_weight_exists": best_weight_path.exists(),
        "last_weight_exists": last_weight_path.exists(),
        "args_yaml_path": args_yaml_path,
        "args_yaml_exists": args_yaml_path is not None and args_yaml_path.exists(),
        "config_files_recorded": config_files_recorded,
        "listed_plot_and_image_files": listed_plot_and_image_files,
    }


def parse_results_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_metrics_info("results.csv missing")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized_field_map = {normalize_column_name(name): name for name in fieldnames}
        rows = list(reader)

    if not fieldnames or not rows:
        return empty_metrics_info("results.csv has no fieldnames or no data rows")

    final_index = len(rows) - 1
    final_row = rows[final_index]
    metric_columns_detected = [name.strip() for name in fieldnames]
    selection_metric = select_best_epoch_metric(normalized_field_map)
    best_index = None
    best_row: dict[str, str] | None = None
    if selection_metric is not None:
        best_index, best_row = find_best_metric_row(rows, selection_metric)

    final_metrics = extract_numeric_metrics(final_row, fieldnames)
    best_metrics = extract_numeric_metrics(best_row, fieldnames) if best_row is not None else {}
    final_epoch_value = coerce_number(final_row.get(normalized_field_map.get("epoch", "epoch"), ""))
    best_epoch_value = None
    if best_row is not None:
        best_epoch_value = coerce_number(best_row.get(normalized_field_map.get("epoch", "epoch"), ""))

    return {
        "parsed": True,
        "parse_note": "",
        "row_count": len(rows),
        "metric_columns_detected": metric_columns_detected,
        "best_epoch_selection_metric": selection_metric,
        "best_epoch_index": best_index,
        "best_epoch_value": best_epoch_value,
        "final_epoch_index": final_index,
        "final_epoch_value": final_epoch_value,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
    }


def empty_metrics_info(note: str) -> dict[str, Any]:
    return {
        "parsed": False,
        "parse_note": note,
        "row_count": 0,
        "metric_columns_detected": [],
        "best_epoch_selection_metric": "",
        "best_epoch_index": None,
        "best_epoch_value": None,
        "final_epoch_index": None,
        "final_epoch_value": None,
        "final_metrics": {},
        "best_metrics": {},
    }


def normalize_column_name(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def select_best_epoch_metric(normalized_field_map: dict[str, str]) -> str:
    priority = ["metrics/map50-95(b)", "metrics/map50(b)", "fitness"]
    for candidate in priority:
        if candidate in normalized_field_map:
            return normalized_field_map[candidate].strip()
    return ""


def find_best_metric_row(rows: list[dict[str, str]], metric_name: str) -> tuple[int | None, dict[str, str] | None]:
    best_index: int | None = None
    best_value: float | None = None
    best_row: dict[str, str] | None = None
    for index, row in enumerate(rows):
        value = coerce_number(row.get(metric_name, ""))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_index = index
            best_row = row
    return best_index, best_row


def extract_numeric_metrics(row: dict[str, str] | None, fieldnames: list[str]) -> dict[str, float | int | str]:
    if row is None:
        return {}
    metrics: dict[str, float | int | str] = {}
    for field in fieldnames:
        raw_value = row.get(field, "")
        parsed = coerce_number(raw_value)
        if parsed is not None:
            metrics[field.strip()] = parsed
        elif raw_value != "":
            metrics[field.strip()] = raw_value
    return metrics


def coerce_number(value: Any) -> float | int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if parsed.is_integer():
        return int(parsed)
    return parsed


def build_weight_manifest_row(weight_label: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    size_bytes = stat.st_size if stat else 0
    return {
        "weight_label": weight_label,
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 6) if exists else 0,
        "sha256": sha256_file(path) if exists else "",
        "modified_time_utc": format_mtime_utc(stat.st_mtime) if stat else "",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def format_mtime_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def build_output_tree(training_output_dir: Path) -> dict[str, Any]:
    if not training_output_dir.exists():
        return {
            "text": f"# Missing training output directory\n{training_output_dir}\n",
            "truncated": False,
        }

    lines = [f"# Phase 11K training output tree", f"# root: {training_output_dir}"]
    count = 0
    truncated = False
    for path in sorted(training_output_dir.rglob("*")):
        relative = path.relative_to(training_output_dir)
        if path.is_dir():
            entry = f"{relative}/"
        else:
            entry = f"{relative} [{path.stat().st_size} bytes]"
        lines.append(entry)
        count += 1
        if count >= TREE_MAX_LINES:
            truncated = True
            lines.append(f"... truncated after {TREE_MAX_LINES} entries ...")
            break
    return {"text": "\n".join(lines) + "\n", "truncated": truncated}


def build_metrics_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": summary["phase"],
        "status": summary["status"],
        "training_output_dir": summary["training_output_dir"],
        "results_csv_path": summary["results_csv_path"],
        "results_row_count": summary["results_row_count"],
        "metric_columns_detected": json.dumps(summary["metric_columns_detected"], ensure_ascii=True),
        "best_epoch_selection_metric": summary["best_epoch_selection_metric"],
        "best_epoch_index": summary["best_epoch_index"],
        "best_epoch_value": summary["best_epoch_value"],
        "final_epoch_index": summary["final_epoch_index"],
        "final_epoch_value": summary["final_epoch_value"],
        "final_metrics_json": json.dumps(summary["final_metrics"], sort_keys=True, ensure_ascii=True),
        "best_metrics_json": json.dumps(summary["best_metrics"], sort_keys=True, ensure_ascii=True),
    }


def build_non_execution_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": summary["status"],
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "dataset_mutated": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated": False,
        "kaggle_upload_executed": False,
        "weights_created_by_phase11k": False,
        "weights_copied_by_phase11k": False,
        "statements": [
            "Phase 11K did not run training.",
            "Phase 11K did not run evaluation.",
            "Phase 11K did not run inference.",
            "Phase 11K did not mutate any dataset.",
            "Phase 11K did not upload anything to Kaggle.",
            "Phase 11K did not create weights/checkpoints.",
            "Phase 11K only inspected existing Phase 11J.1 output files and wrote small metadata artifacts.",
        ],
        "inspected_training_output_dir": summary["training_output_dir"],
        "generated_at_utc": summary["generated_at_utc"],
    }


def build_readme(summary: dict[str, Any], weight_rows: list[dict[str, Any]], tree_info: dict[str, Any]) -> str:
    lines = [
        "# Phase 11K Training Outputs And Metrics",
        "",
        "Phase 11K is a strict non-execution provenance phase.",
        "",
        f"- status = `{summary['status']}`",
        f"- phase11j1_summary_available = `{summary['phase11j1_summary_available']}`",
        f"- phase11j1_status = `{summary['phase11j1_status']}`",
        f"- training_output_dir = `{summary['training_output_dir']}`",
        f"- results_csv_exists = `{summary['results_csv_exists']}`",
        f"- best_weight_exists = `{summary['best_weight_exists']}`",
        f"- last_weight_exists = `{summary['last_weight_exists']}`",
        f"- results_row_count = `{summary['results_row_count']}`",
        f"- results_csv_parsed = `{summary['results_csv_parsed']}`",
        f"- best_epoch_selection_metric = `{summary['best_epoch_selection_metric']}`",
        f"- best_epoch_index = `{summary['best_epoch_index']}`",
        f"- final_epoch_index = `{summary['final_epoch_index']}`",
        f"- weights_hashed = `{summary['weights_hashed']}`",
        f"- next_allowed_step = `{summary['next_allowed_step']}`",
        "",
        "Non-execution guarantees:",
        "",
        "- Phase 11K did not run training.",
        "- Phase 11K did not run evaluation.",
        "- Phase 11K did not run inference.",
        "- Phase 11K did not mutate any dataset.",
        "- Phase 11K did not upload anything to Kaggle.",
        "- Phase 11K did not create or copy weights/checkpoints.",
        "",
        "Weight files recorded by path, size, and sha256 only:",
        "",
    ]
    for row in weight_rows:
        lines.append(
            f"- `{row['weight_label']}`: exists=`{row['exists']}`, size_bytes=`{row['size_bytes']}`, sha256=`{row['sha256']}`"
        )
    lines.extend(
        [
            "",
            "Training output tree artifact:",
            "",
            f"- truncated = `{tree_info['truncated']}`",
            f"- path = `{summary['training_output_tree_path']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


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


def path_or_empty(path: Path | None) -> str:
    return str(path) if path is not None else ""


if __name__ == "__main__":
    main()
