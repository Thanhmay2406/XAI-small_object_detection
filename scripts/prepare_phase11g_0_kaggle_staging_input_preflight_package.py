"""Prepare the Phase 11G.0 Kaggle staging-input preflight package without execution."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "11G.0"
STATUS = "phase11g_0_kaggle_staging_input_preflight_package_prepared"
NEXT_ALLOWED_STEP = "upload_phase9s_staging_dataset_copy_to_kaggle_and_run_phase11g_preflight"
REPO_ROOT_REQUIRED_DIRS = ("artifacts", "docs", "scripts", "src")
OUTPUT_DIR_DEFAULT = "artifacts/phase11g_0_kaggle_staging_input_preflight_package"
PHASE11F_SUMMARY_REL = (
    "artifacts/phase11f_approved_staging_training_execution_package/phase11f_execution_package_summary.json"
)
PHASE11F_1_SUMMARY_REL = (
    "artifacts/phase11f_1_kaggle_execution_adapter_package/phase11f_1_kaggle_adapter_summary.json"
)
STAGING_DATASET_COPY_REL = (
    "artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_copy"
)
STAGING_YAML_REL = (
    "artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml"
)
EXPECTED_PHASE11F_STATUS = "phase11f_approved_staging_training_execution_package_prepared"
EXPECTED_PHASE11F_1_STATUS = "phase11f_1_kaggle_execution_adapter_package_prepared"
EXPECTED_PHASE11F_1_NEXT = "phase11g_execute_approved_staging_training_on_kaggle"
KAGGLE_YAML_PATH = "/kaggle/input/REPLACE_WITH_STAGING_DATASET_SLUG/staging_dataset_copy"
KAGGLE_REPO_ROOT_DEFAULT = "/kaggle/working/XAI-small_object_detection"
KAGGLE_DATA_YAML_DEFAULT = "/kaggle/working/staging_dataset_drill_bit_yolo_kaggle.yaml"
KAGGLE_PROJECT_DIR_DEFAULT = "/kaggle/working/experiments/phase11g_staging_training_chipped"
CLASS_NAMES = {
    0: "Broken",
    1: "Chipped",
    2: "Scratched",
    3: "Severe_Rust",
    4: "Tip_Wear",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Phase 11G.0 Kaggle staging-input preflight package without training."
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
    phase11f_1_summary = read_json_required(
        repo_root / PHASE11F_1_SUMMARY_REL, "Phase 11F.1 summary"
    )
    phase11f_ready, phase11f_failures = validate_phase11f_summary(phase11f_summary)
    phase11f_1_ready, phase11f_1_failures = validate_phase11f_1_summary(phase11f_1_summary)

    staging_dataset_copy = repo_root / STAGING_DATASET_COPY_REL
    staging_yaml = repo_root / STAGING_YAML_REL
    dataset_exists = staging_dataset_copy.exists()
    yaml_exists = staging_yaml.exists()
    local_yaml_text = read_optional_text(staging_yaml)
    local_yaml_has_absolute_path = "/home/thanhmay/" in local_yaml_text

    split_inventory = collect_split_inventory(staging_dataset_copy)
    directories_ready = all(row["directory_exists"] for row in split_inventory)
    ready_for_upload = phase11f_ready and phase11f_1_ready and dataset_exists and yaml_exists and directories_ready

    package = build_package(
        phase11f_summary=phase11f_summary,
        phase11f_1_summary=phase11f_1_summary,
        phase11f_ready=phase11f_ready,
        phase11f_1_ready=phase11f_1_ready,
        phase11f_failures=phase11f_failures,
        phase11f_1_failures=phase11f_1_failures,
        split_inventory=split_inventory,
        staging_dataset_copy_exists=dataset_exists,
        staging_yaml_exists=yaml_exists,
        local_yaml_has_absolute_path=local_yaml_has_absolute_path,
        ready_for_upload=ready_for_upload,
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


def read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def validate_phase11f_summary(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if summary.get("status") != EXPECTED_PHASE11F_STATUS:
        failures.append(f"status must equal '{EXPECTED_PHASE11F_STATUS}'")
    if summary.get("training_executed") is not False:
        failures.append("training_executed must equal false")
    if summary.get("training_command_executed") is not False:
        failures.append("training_command_executed must equal false")
    return not failures, failures


def validate_phase11f_1_summary(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if summary.get("status") != EXPECTED_PHASE11F_1_STATUS:
        failures.append(f"status must equal '{EXPECTED_PHASE11F_1_STATUS}'")
    if summary.get("next_allowed_step") != EXPECTED_PHASE11F_1_NEXT:
        failures.append(f"next_allowed_step must equal '{EXPECTED_PHASE11F_1_NEXT}'")
    if summary.get("training_executed") is not False:
        failures.append("training_executed must equal false")
    return not failures, failures


def collect_split_inventory(staging_dataset_copy: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        image_dir = staging_dataset_copy / "images" / split
        label_dir = staging_dataset_copy / "labels" / split
        image_count = count_entries(image_dir, kind="image")
        label_count = count_entries(label_dir, kind="label")
        rows.append(
            {
                "split": split,
                "image_dir": image_dir,
                "label_dir": label_dir,
                "image_dir_exists": image_dir.is_dir(),
                "label_dir_exists": label_dir.is_dir(),
                "directory_exists": image_dir.is_dir() and label_dir.is_dir(),
                "image_count": image_count,
                "label_count": label_count,
                "count_difference": image_count - label_count,
            }
        )
    return rows


def count_entries(directory: Path, kind: str) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for path in directory.iterdir():
        if not (path.is_file() or path.is_symlink()):
            continue
        if kind == "image" and path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
        elif kind == "label" and path.suffix.lower() == ".txt":
            count += 1
    return count


def build_package(
    phase11f_summary: dict[str, Any],
    phase11f_1_summary: dict[str, Any],
    phase11f_ready: bool,
    phase11f_1_ready: bool,
    phase11f_failures: list[str],
    phase11f_1_failures: list[str],
    split_inventory: list[dict[str, Any]],
    staging_dataset_copy_exists: bool,
    staging_yaml_exists: bool,
    local_yaml_has_absolute_path: bool,
    ready_for_upload: bool,
) -> dict[str, Any]:
    summary = {
        "phase": PHASE,
        "status": STATUS,
        "source_phase11f_ready": phase11f_ready,
        "source_phase11f_1_ready": phase11f_1_ready,
        "staging_dataset_copy_exists": staging_dataset_copy_exists,
        "staging_yaml_exists": staging_yaml_exists,
        "local_staging_yaml_has_local_absolute_path": local_yaml_has_absolute_path,
        "kaggle_yaml_template_prepared": True,
        "kaggle_notebook_commands_prepared": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated_by_phase11g_0": False,
        "ready_for_kaggle_upload_of_staging_dataset": ready_for_upload,
        "ready_for_phase11g_training_after_kaggle_preflight": False,
        "next_allowed_step": NEXT_ALLOWED_STEP,
        "phase11f_summary_path": PHASE11F_SUMMARY_REL,
        "phase11f_1_summary_path": PHASE11F_1_SUMMARY_REL,
        "staging_dataset_copy_path": STAGING_DATASET_COPY_REL,
        "staging_yaml_path": STAGING_YAML_REL,
        "phase11f_guardrail_failures": phase11f_failures,
        "phase11f_1_guardrail_failures": phase11f_1_failures,
        "split_inventory": [
            {
                "split": row["split"],
                "image_count": row["image_count"],
                "label_count": row["label_count"],
                "count_difference": row["count_difference"],
                "directories_ready": row["directory_exists"],
            }
            for row in split_inventory
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "summary": summary,
        "inventory_rows": build_inventory_rows(split_inventory),
        "yaml_template": build_kaggle_yaml_template(),
        "notebook_commands": build_notebook_commands(),
        "guardrail_rows": build_guardrail_rows(
            phase11f_summary=phase11f_summary,
            phase11f_1_summary=phase11f_1_summary,
            phase11f_ready=phase11f_ready,
            phase11f_1_ready=phase11f_1_ready,
            phase11f_failures=phase11f_failures,
            phase11f_1_failures=phase11f_1_failures,
            split_inventory=split_inventory,
            local_yaml_has_absolute_path=local_yaml_has_absolute_path,
        ),
        "non_execution_manifest": build_non_execution_manifest(),
        "readme_text": build_readme(summary, split_inventory),
    }


def build_inventory_rows(split_inventory: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in split_inventory:
        rows.append(
            {
                "split": row["split"],
                "image_dir_exists": str(row["image_dir_exists"]).lower(),
                "label_dir_exists": str(row["label_dir_exists"]).lower(),
                "image_count": str(row["image_count"]),
                "label_count": str(row["label_count"]),
                "count_difference": str(row["count_difference"]),
                "notes": "images may be symlinks; counts include symlinked image entries",
            }
        )
    return rows


def build_kaggle_yaml_template() -> str:
    lines = [
        f"path: {KAGGLE_YAML_PATH}",
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for class_id, class_name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {class_name}")
    lines.append("")
    return "\n".join(lines)


def build_notebook_commands() -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# Phase 11G.0 Kaggle preflight only. Do not run training from this script.",
        "",
        "set -eu",
        "",
        "cd /kaggle/working",
        "git clone https://github.com/REPLACE_WITH_OWNER/XAI-small_object_detection.git",
        "cd XAI-small_object_detection",
        "# If a newer commit exists that includes Phase 11G.0, replace the hash below with that commit.",
        "git checkout 1fbac0d7cf18e96a6e3ce852139305b1ffc577b9",
        "# Example for a newer Phase 11G.0 commit:",
        "# git checkout REPLACE_WITH_PHASE11G_0_COMMIT",
        "",
        f"export REPO_ROOT={KAGGLE_REPO_ROOT_DEFAULT}",
        f"export DATA_YAML={KAGGLE_DATA_YAML_DEFAULT}",
        f"export PROJECT_DIR={KAGGLE_PROJECT_DIR_DEFAULT}",
        "",
        "cat > \"$DATA_YAML\" <<'YAML'",
        build_kaggle_yaml_template().rstrip(),
        "YAML",
        "",
        "test -f \"$DATA_YAML\"",
        "cat \"$DATA_YAML\"",
        "find /kaggle/input -maxdepth 3 -type d | sort | head -100",
        "python - <<'PY'",
        "from pathlib import Path",
        "import json",
        "import os",
        "",
        "try:",
        "    import yaml",
        "except ImportError as exc:",
        "    raise SystemExit(f'PyYAML is required for preflight: {exc}')",
        "",
        "data_yaml = Path(os.environ['DATA_YAML'])",
        "payload = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))",
        "root = Path(payload['path'])",
        "report = {'data_yaml': str(data_yaml), 'path': str(root), 'splits': {}}",
        "for split_key, split_name in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:",
        "    rel = payload[split_key]",
        "    image_dir = root / rel",
        "    label_dir = root / 'labels' / split_name",
        "    images = sorted(p for p in image_dir.iterdir()) if image_dir.is_dir() else []",
        "    labels = sorted(p for p in label_dir.iterdir()) if label_dir.is_dir() else []",
        "    report['splits'][split_name] = {",
        "        'image_dir_exists': image_dir.is_dir(),",
        "        'label_dir_exists': label_dir.is_dir(),",
        "        'image_count': len([p for p in images if p.is_file() or p.is_symlink()]),",
        "        'label_count': len([p for p in labels if p.is_file() or p.is_symlink()]),",
        "    }",
        "try:",
        "    import torch",
        "except ImportError:",
        "    report['torch_cuda_available'] = 'torch_not_installed'",
        "else:",
        "    report['torch_cuda_available'] = bool(torch.cuda.is_available())",
        "print(json.dumps(report, indent=2))",
        "PY",
        "",
        "# Phase 11G training command is intentionally not executed in this preflight script.",
        "# Example only; do not uncomment until Kaggle preflight has passed:",
        "# yolo detect train data=\"$DATA_YAML\" model=yolov8n.pt epochs=100 imgsz=640 batch=16 seed=42 project=\"$PROJECT_DIR\" name=yolov8n_staging_chipped_phase11g",
        "",
    ]
    return "\n".join(lines)


def build_guardrail_rows(
    phase11f_summary: dict[str, Any],
    phase11f_1_summary: dict[str, Any],
    phase11f_ready: bool,
    phase11f_1_ready: bool,
    phase11f_failures: list[str],
    phase11f_1_failures: list[str],
    split_inventory: list[dict[str, Any]],
    local_yaml_has_absolute_path: bool,
) -> list[dict[str, str]]:
    inventory_ready = all(row["directory_exists"] for row in split_inventory)
    return [
        guardrail_row(
            "phase11f_status_and_non_execution",
            "pass" if phase11f_ready else "fail",
            phase11f_summary.get("status", ""),
            "; ".join(phase11f_failures) or "Phase 11F status and no-training guardrail satisfied.",
        ),
        guardrail_row(
            "phase11f_1_status_and_next_step",
            "pass" if phase11f_1_ready else "fail",
            phase11f_1_summary.get("status", ""),
            "; ".join(phase11f_1_failures)
            or "Phase 11F.1 status, next step, and no-training guardrail satisfied.",
        ),
        guardrail_row(
            "staging_dataset_layout",
            "pass" if inventory_ready else "fail",
            STAGING_DATASET_COPY_REL,
            "All required images/* and labels/* split directories exist."
            if inventory_ready
            else "One or more required split directories are missing.",
        ),
        guardrail_row(
            "local_yaml_absolute_path_detection",
            "warn" if local_yaml_has_absolute_path else "pass",
            STAGING_YAML_REL,
            "Local staging YAML contains a /home/thanhmay absolute path and must be rewritten on Kaggle."
            if local_yaml_has_absolute_path
            else "Local staging YAML is already free of local absolute paths.",
        ),
        guardrail_row(
            "phase11g_0_execution_boundary",
            "pass",
            PHASE,
            "Phase 11G.0 prepares and checks inputs only. Training, evaluation, and inference remain unexecuted.",
        ),
    ]


def build_non_execution_manifest() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "status": STATUS,
        "prepare_only": True,
        "training_executed": False,
        "evaluation_executed": False,
        "inference_executed": False,
        "original_dataset_mutated": False,
        "staging_dataset_mutated_by_phase11g_0": False,
        "dataset_labels_mutated": False,
        "dataset_images_mutated": False,
        "kaggle_preflight_executed_locally": False,
        "training_command_executed": False,
        "notes": "Phase 11G.0 prepares Kaggle staging input and preflight materials only.",
    }


def build_readme(summary: dict[str, Any], split_inventory: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 11G.0 Kaggle Staging Input Preflight Package",
        "",
        "This package prepares Kaggle-side staging-input checks only. It does not run training, evaluation, or inference.",
        "",
        "## Why this package exists",
        "- Phase 11G must use the Phase 9S staging dataset copy, not the original `data/` dataset tree.",
        "- The local staging YAML contains a local absolute path and cannot be used on Kaggle without rewriting.",
        "- Kaggle preflight must pass before any real Phase 11G training command is allowed to run.",
        "",
        "## Local staging inventory",
    ]
    for row in split_inventory:
        lines.append(
            f"- `{row['split']}`: images={row['image_count']}, labels={row['label_count']}, directories_ready={str(row['directory_exists']).lower()}"
        )
    lines.extend(
        [
            "",
            "## Summary",
            f"- status: `{summary['status']}`",
            f"- ready_for_kaggle_upload_of_staging_dataset: `{str(summary['ready_for_kaggle_upload_of_staging_dataset']).lower()}`",
            "- ready_for_phase11g_training_after_kaggle_preflight: `false`",
            f"- next_allowed_step: `{summary['next_allowed_step']}`",
            "",
            "## Files",
            "- `phase11g_0_preflight_summary.json`",
            "- `phase11g_0_staging_dataset_inventory.csv`",
            "- `phase11g_0_kaggle_yaml_template.yaml`",
            "- `phase11g_0_kaggle_notebook_commands.sh`",
            "- `phase11g_0_guardrail_checks.csv`",
            "- `phase11g_0_non_execution_manifest.json`",
            "- `README.md`",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, package: dict[str, Any]) -> None:
    write_json(output_dir / "phase11g_0_preflight_summary.json", package["summary"])
    write_csv(
        output_dir / "phase11g_0_staging_dataset_inventory.csv",
        package["inventory_rows"],
        fieldnames=[
            "split",
            "image_dir_exists",
            "label_dir_exists",
            "image_count",
            "label_count",
            "count_difference",
            "notes",
        ],
    )
    (output_dir / "phase11g_0_kaggle_yaml_template.yaml").write_text(
        package["yaml_template"], encoding="utf-8"
    )
    notebook_path = output_dir / "phase11g_0_kaggle_notebook_commands.sh"
    notebook_path.write_text(package["notebook_commands"], encoding="utf-8")
    notebook_path.chmod(0o755)
    write_csv(
        output_dir / "phase11g_0_guardrail_checks.csv",
        package["guardrail_rows"],
        fieldnames=["check_id", "status", "evidence", "notes"],
    )
    write_json(
        output_dir / "phase11g_0_non_execution_manifest.json",
        package["non_execution_manifest"],
    )
    (output_dir / "README.md").write_text(package["readme_text"], encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def guardrail_row(check_id: str, status: str, evidence: str, notes: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "status": status,
        "evidence": evidence,
        "notes": notes,
    }


if __name__ == "__main__":
    main()
