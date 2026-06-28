"""Materialize the Phase 11H Kaggle manual preflight validation gate package."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "11H"
SOURCE_PHASE = "11H_manual_kaggle_preflight_validation"
STATUS = "phase11h_kaggle_manual_preflight_validation_passed"
NEXT_ALLOWED_STEP = "phase11i_human_training_execution_approval_gate"
REPO_ROOT_REQUIRED_DIRS = ("artifacts", "docs", "scripts", "src")
OUTPUT_DIR_DEFAULT = "artifacts/phase11h_kaggle_manual_preflight_validation"
INPUT_SUMMARY_DEFAULT = (
    "artifacts/phase11h_kaggle_manual_preflight_validation/input/"
    "phase11h_manual_kaggle_preflight_validation_summary.json"
)
EXPECTED_SPLITS = ("train", "valid", "test")
EXPECTED_YAML_SPLITS = {
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
}
EXPECTED_CLASS_NAMES = [
    ("0", "Broken"),
    ("1", "Chipped"),
    ("2", "Scratched"),
    ("3", "Severe_Rust"),
    ("4", "Tip_Wear"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the Phase 11H Kaggle manual preflight validation gate package."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--input-summary", default=INPUT_SUMMARY_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    input_summary_path = (repo_root / args.input_summary).resolve()

    ensure_repo_root(repo_root)
    summary = read_json_required(input_summary_path, "Phase 11H manual preflight summary")
    validate_input_summary(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    package = build_package(
        repo_root=repo_root,
        output_dir=output_dir,
        input_summary=summary,
        input_summary_path=input_summary_path,
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


def validate_input_summary(summary: dict[str, Any]) -> None:
    failures: list[str] = []
    check_equal(summary, "phase", SOURCE_PHASE, failures)
    check_equal(summary, "mode", "gate_only_no_training", failures)
    check_equal(summary, "dataset_root_exists", True, failures)
    check_equal(summary, "yaml_exists", True, failures)
    check_equal(summary, "manual_kaggle_preflight_structural_passed", True, failures)
    check_equal(summary, "ready_for_training_execution_candidate", True, failures)
    check_equal(summary, "training_execution_approval_required", True, failures)
    check_equal(summary, "errors", [], failures)
    check_equal(summary, "warnings", [], failures)
    check_equal(summary, "training_executed", False, failures)
    check_equal(summary, "evaluation_executed", False, failures)
    check_equal(summary, "inference_executed", False, failures)
    check_equal(summary, "ready_for_training_execution", False, failures)

    validate_splits(summary.get("splits"), failures)
    validate_yaml(summary.get("yaml"), failures)

    if failures:
        raise SystemExit("Phase 11H input summary validation failed:\n- " + "\n- ".join(failures))


def check_equal(summary: dict[str, Any], field: str, expected: Any, failures: list[str]) -> None:
    actual = summary.get(field)
    if actual != expected:
        failures.append(f"{field} must equal {expected!r}; got {actual!r}")


def validate_splits(splits: Any, failures: list[str]) -> None:
    if not isinstance(splits, dict):
        failures.append("splits must be a JSON object")
        return
    for split_name in EXPECTED_SPLITS:
        payload = splits.get(split_name)
        if not isinstance(payload, dict):
            failures.append(f"splits.{split_name} must be a JSON object")
            continue
        for flag_name in ("image_dir_exists", "label_dir_exists"):
            if payload.get(flag_name) is not True:
                failures.append(f"splits.{split_name}.{flag_name} must be true")
        image_count = payload.get("image_count")
        label_count = payload.get("label_count")
        if not isinstance(image_count, int) or image_count <= 0:
            failures.append(f"splits.{split_name}.image_count must be an integer > 0")
        if not isinstance(label_count, int) or label_count <= 0:
            failures.append(f"splits.{split_name}.label_count must be an integer > 0")
        if image_count != label_count:
            failures.append(
                f"splits.{split_name}.image_count must equal label_count; got {image_count!r} vs {label_count!r}"
            )
        for zero_name in ("missing_label_count", "orphan_label_count", "invalid_label_count"):
            if payload.get(zero_name) != 0:
                failures.append(f"splits.{split_name}.{zero_name} must equal 0")


def validate_yaml(yaml_payload: Any, failures: list[str]) -> None:
    if not isinstance(yaml_payload, dict):
        failures.append("yaml must be a JSON object")
        return
    for key, expected in EXPECTED_YAML_SPLITS.items():
        actual = yaml_payload.get(key)
        if actual != expected:
            failures.append(f"yaml.{key} must equal {expected!r}; got {actual!r}")
    if yaml_payload.get("nc") != 5:
        failures.append(f"yaml.nc must equal 5; got {yaml_payload.get('nc')!r}")

    names_payload = yaml_payload.get("names")
    if not isinstance(names_payload, dict):
        failures.append("yaml.names must be a JSON object")
        return
    actual_names = [(str(key), str(value)) for key, value in names_payload.items()]
    if actual_names != EXPECTED_CLASS_NAMES:
        failures.append(
            "yaml.names must preserve the expected ordered mapping "
            f"{EXPECTED_CLASS_NAMES!r}; got {actual_names!r}"
        )


def build_package(
    repo_root: Path,
    output_dir: Path,
    input_summary: dict[str, Any],
    input_summary_path: Path,
) -> dict[str, Any]:
    input_summary_rel = normalize_relpath(repo_root, input_summary_path)
    split_counts_rows = build_split_counts_rows(input_summary["splits"])
    class_counts_rows = build_class_counts_rows(input_summary["splits"])
    gate_decision = build_gate_decision(input_summary, input_summary_rel)
    non_execution_manifest = build_non_execution_manifest(input_summary_rel)
    summary = build_summary(
        input_summary=input_summary,
        input_summary_rel=input_summary_rel,
        split_counts_rows=split_counts_rows,
        class_counts_rows=class_counts_rows,
    )
    readme_text = build_readme(summary, gate_decision, split_counts_rows)
    return {
        "summary": summary,
        "split_counts_rows": split_counts_rows,
        "class_counts_rows": class_counts_rows,
        "gate_decision": gate_decision,
        "non_execution_manifest": non_execution_manifest,
        "readme_text": readme_text,
    }


def build_split_counts_rows(splits: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_image_count = 0
    total_label_count = 0
    total_label_line_count = 0
    total_empty_label_file_count = 0
    total_missing_label_count = 0
    total_orphan_label_count = 0
    total_invalid_label_count = 0
    for split_name in EXPECTED_SPLITS:
        payload = splits[split_name]
        row = {
            "split": split_name,
            "image_dir": payload["image_dir"],
            "label_dir": payload["label_dir"],
            "image_count": payload["image_count"],
            "label_count": payload["label_count"],
            "label_line_count": payload.get("label_line_count", 0),
            "empty_label_file_count": payload.get("empty_label_file_count", 0),
            "missing_label_count": payload["missing_label_count"],
            "orphan_label_count": payload["orphan_label_count"],
            "invalid_label_count": payload["invalid_label_count"],
        }
        rows.append(row)
        total_image_count += int(row["image_count"])
        total_label_count += int(row["label_count"])
        total_label_line_count += int(row["label_line_count"])
        total_empty_label_file_count += int(row["empty_label_file_count"])
        total_missing_label_count += int(row["missing_label_count"])
        total_orphan_label_count += int(row["orphan_label_count"])
        total_invalid_label_count += int(row["invalid_label_count"])
    rows.append(
        {
            "split": "total",
            "image_dir": "",
            "label_dir": "",
            "image_count": total_image_count,
            "label_count": total_label_count,
            "label_line_count": total_label_line_count,
            "empty_label_file_count": total_empty_label_file_count,
            "missing_label_count": total_missing_label_count,
            "orphan_label_count": total_orphan_label_count,
            "invalid_label_count": total_invalid_label_count,
        }
    )
    return rows


def build_class_counts_rows(splits: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    totals_by_class = {class_id: 0 for class_id, _ in EXPECTED_CLASS_NAMES}
    for split_name in EXPECTED_SPLITS:
        class_counts = splits[split_name].get("class_counts", {})
        split_total = 0
        for class_id, class_name in EXPECTED_CLASS_NAMES:
            count = int(class_counts.get(class_id, 0))
            rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "count": count,
                }
            )
            totals_by_class[class_id] += count
            split_total += count
        rows.append(
            {
                "split": split_name,
                "class_id": "all",
                "class_name": "all",
                "count": split_total,
            }
        )
    grand_total = 0
    for class_id, class_name in EXPECTED_CLASS_NAMES:
        count = totals_by_class[class_id]
        rows.append(
            {
                "split": "total",
                "class_id": class_id,
                "class_name": class_name,
                "count": count,
            }
        )
        grand_total += count
    rows.append(
        {
            "split": "total",
            "class_id": "all",
            "class_name": "all",
            "count": grand_total,
        }
    )
    return rows


def build_gate_decision(input_summary: dict[str, Any], input_summary_rel: str) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": STATUS,
        "source_summary_path": input_summary_rel,
        "dataset_root": input_summary["dataset_root"],
        "yaml_path": input_summary["yaml_path"],
        "manual_kaggle_preflight_structural_passed": True,
        "ready_for_training_execution_candidate": True,
        "ready_for_training_execution": False,
        "training_execution_approval_required": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated": False,
        "kaggle_dataset_uploaded": False,
        "weights_or_checkpoints_created": False,
        "next_allowed_step": NEXT_ALLOWED_STEP,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_non_execution_manifest(input_summary_rel: str) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": STATUS,
        "source_summary_path": input_summary_rel,
        "no_training": True,
        "no_evaluation": True,
        "no_inference": True,
        "no_dataset_mutation": True,
        "no_kaggle_upload": True,
        "no_weights_or_checkpoints": True,
        "notes": "Phase 11H materializes a manual Kaggle preflight pass only. It does not execute or mutate anything.",
    }


def build_summary(
    input_summary: dict[str, Any],
    input_summary_rel: str,
    split_counts_rows: list[dict[str, Any]],
    class_counts_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_row = next(row for row in split_counts_rows if row["split"] == "total")
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "status": STATUS,
        "mode": "gate_only_record_only_no_training",
        "source_summary_path": input_summary_rel,
        "dataset_root": input_summary["dataset_root"],
        "yaml_path": input_summary["yaml_path"],
        "manual_kaggle_preflight_structural_passed": True,
        "ready_for_training_execution_candidate": True,
        "ready_for_training_execution": False,
        "training_execution_approval_required": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated": False,
        "kaggle_dataset_uploaded": False,
        "weights_or_checkpoints_created": False,
        "yaml_validation_passed": True,
        "split_validation_passed": True,
        "split_count_rows": len(split_counts_rows),
        "class_count_rows": len(class_counts_rows),
        "total_images": total_row["image_count"],
        "total_labels": total_row["label_count"],
        "total_label_lines": total_row["label_line_count"],
        "total_empty_label_files": total_row["empty_label_file_count"],
        "total_missing_labels": total_row["missing_label_count"],
        "total_orphan_labels": total_row["orphan_label_count"],
        "total_invalid_labels": total_row["invalid_label_count"],
        "next_allowed_step": NEXT_ALLOWED_STEP,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "summary_json": "phase11h_kaggle_manual_preflight_validation_summary.json",
            "split_counts_csv": "phase11h_split_counts.csv",
            "class_counts_csv": "phase11h_class_counts.csv",
            "gate_decision_json": "phase11h_gate_decision.json",
            "non_execution_manifest_json": "phase11h_non_execution_manifest.json",
            "readme_md": "README.md",
        },
    }


def build_readme(
    summary: dict[str, Any],
    gate_decision: dict[str, Any],
    split_counts_rows: list[dict[str, Any]],
) -> str:
    split_lines = []
    for row in split_counts_rows:
        if row["split"] == "total":
            continue
        split_lines.append(
            f"- {row['split']}: images={row['image_count']}, labels={row['label_count']}, "
            f"missing={row['missing_label_count']}, orphan={row['orphan_label_count']}, invalid={row['invalid_label_count']}"
        )
    split_lines.extend(
        [
            f"- total_images = `{summary['total_images']}`",
            f"- total_labels = `{summary['total_labels']}`",
            f"- total_label_lines = `{summary['total_label_lines']}`",
            f"- total_empty_label_files = `{summary['total_empty_label_files']}`",
        ]
    )
    lines = [
        "# Phase 11H Kaggle Manual Preflight Validation",
        "",
        "Phase 11H materializes the copied Kaggle manual preflight summary as a gate-only, record-only artifact bundle.",
        "",
        "## Outcome",
        f"- status = `{summary['status']}`",
        f"- source_summary_path = `{summary['source_summary_path']}`",
        f"- manual_kaggle_preflight_structural_passed = `{summary['manual_kaggle_preflight_structural_passed']}`",
        f"- ready_for_training_execution_candidate = `{summary['ready_for_training_execution_candidate']}`",
        f"- ready_for_training_execution = `{summary['ready_for_training_execution']}`",
        f"- training_execution_approval_required = `{summary['training_execution_approval_required']}`",
        f"- training_executed = `{summary['training_executed']}`",
        f"- evaluation_executed = `{summary['evaluation_executed']}`",
        f"- inference_executed = `{summary['inference_executed']}`",
        f"- next_allowed_step = `{gate_decision['next_allowed_step']}`",
        "",
        "## Split Counts",
        *split_lines,
        "",
        "## Guardrails",
        "- No training, evaluation, or inference is executed in Phase 11H.",
        "- No original dataset or staging dataset mutation is performed in Phase 11H.",
        "- No Kaggle dataset upload is performed in Phase 11H.",
        "- No weights or checkpoints are created in Phase 11H.",
    ]
    return "\n".join(lines) + "\n"


def normalize_relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_outputs(output_dir: Path, package: dict[str, Any]) -> None:
    write_json(output_dir / "phase11h_kaggle_manual_preflight_validation_summary.json", package["summary"])
    write_csv(output_dir / "phase11h_split_counts.csv", package["split_counts_rows"])
    write_csv(output_dir / "phase11h_class_counts.csv", package["class_counts_rows"])
    write_json(output_dir / "phase11h_gate_decision.json", package["gate_decision"])
    write_json(output_dir / "phase11h_non_execution_manifest.json", package["non_execution_manifest"])
    (output_dir / "README.md").write_text(package["readme_text"], encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
