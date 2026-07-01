"""Preflight-first training entrypoint with a verified runtime weighting hook."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.m8_v1c_runtime import M8V1CPolicyMiningTrainer, M8V1CRuntimeDetectionModel, runtime_policy_from_config_path

OUTPUT_DIR = Path("artifacts/runtime_weighted_training")
SUMMARY_PATH = OUTPUT_DIR / "training_entrypoint_summary.json"
CHECK_PATH = OUTPUT_DIR / "training_entrypoint_check.csv"
COMMAND_DRAFT_PATH = OUTPUT_DIR / "training_command_draft.txt"
MANIFEST_PATH = OUTPUT_DIR / "training_entrypoint_non_execution_manifest.json"
README_PATH = OUTPUT_DIR / "README.md"

SUMMARY_FIELDS = (
    "phase",
    "status",
    "dataset_config_exists",
    "model_config_exists",
    "method_config_exists",
    "method_config_policy_verified",
    "runtime_method_integration_verified",
    "dry_run_passed",
    "execute_flag_required",
    "training_execution_ready",
    "training_allowed",
    "training_executed",
    "evaluation_executed",
    "inference_executed",
    "prediction_executed",
    "dataset_mutated",
    "checkpoint_loaded",
    "checkpoint_mutated",
    "next_allowed_step",
)


@dataclass
class CheckRow:
    item: str
    status: str
    evidence: str
    notes: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or execute the repo-local runtime-weighted YOLO training entrypoint."
    )
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--model", required=True, help="Model YAML path or basename.")
    parser.add_argument("--method-config", required=True, help="Runtime weighting config YAML path.")
    parser.add_argument("--epochs", type=int, required=True, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, required=True, help="Training image size.")
    parser.add_argument("--batch", type=int, required=True, help="Training batch size.")
    parser.add_argument("--device", required=True, help="Training device.")
    parser.add_argument("--workers", type=int, required=True, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, required=True, help="Early stopping patience.")
    parser.add_argument("--project", required=True, help="Ultralytics project directory.")
    parser.add_argument("--name", required=True, help="Run name.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--weights", default=None, help="Optional pretrained weights path.")
    parser.add_argument("--dry-run", action="store_true", help="Explicitly request non-executing preflight mode.")
    parser.add_argument("--execute", action="store_true", help="Allow training execution after all guards pass.")
    return parser


def resolve_existing_path(raw_value: str, *, search_dirs: tuple[Path, ...] = ()) -> tuple[Path, list[Path]]:
    raw_path = Path(raw_value)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(Path.cwd() / raw_path)
        for directory in search_dirs:
            candidates.append(Path.cwd() / directory / raw_value)
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(normalized)
        if normalized.exists():
            return normalized, unique_candidates
    return unique_candidates[0], unique_candidates


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[CheckRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "evidence", "notes"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def validate_dataset_config(path: Path) -> tuple[bool, str, dict[str, Any]]:
    payload = load_yaml_mapping(path)
    has_train = "train" in payload
    has_val = "val" in payload or "valid" in payload
    has_nc = "nc" in payload
    has_names = "names" in payload
    is_valid = has_train and has_val and has_nc and has_names
    notes = f"train={has_train}, val_or_valid={has_val}, nc={has_nc}, names={has_names}"
    return is_valid, notes, payload


def validate_method_policy(path: Path) -> tuple[bool, str]:
    policy = runtime_policy_from_config_path(path)
    issues: list[str] = []
    notes: list[str] = [f"application_mode={policy.application_mode}"]

    raw_payload = load_yaml_mapping(path)
    source_artifact = raw_payload.get("policy_source_artifact")
    if source_artifact is not None and not str(source_artifact).strip():
        issues.append("policy_source_artifact is present but empty")

    preferred_groups: list[str] = []
    identity_groups: list[str] = []
    for group_name in ("small", "medium", "large"):
        group = policy.groups[group_name]
        resolved_max = max(group.weight_delta, key=group.weight_delta.get)
        if group.mode == "preferred_scale":
            preferred_groups.append(group_name)
            notes.append(f"{group_name}=preferred_scale:{group.preferred_scale} (peak={resolved_max})")
        elif group.mode == "identity":
            identity_groups.append(group_name)
            notes.append(f"{group_name}=identity")
        else:
            issues.append(f"{group_name} has unsupported mode {group.mode!r}")

    unknown_weights = policy.weights_for_group("unknown")
    if any(abs(weight - 1.0) > 1e-9 for weight in unknown_weights.values()):
        issues.append(f"unknown must be identity, got {unknown_weights!r}")

    if not preferred_groups:
        notes.append("all size-aware groups currently resolve to identity; runtime is valid but behavior is effectively baseline-like")

    if source_artifact is not None:
        notes.append(f"policy_source_artifact={source_artifact}")

    if issues:
        return False, "; ".join(issues)
    return True, "runtime policy verified: " + "; ".join(notes)


def verify_runtime_integration(
    model_path: Path,
    data_payload: dict[str, Any],
    method_config_path: Path,
) -> tuple[bool, str]:
    channels = int(data_payload.get("channels", 3) or 3)
    nc = int(data_payload["nc"])
    policy = runtime_policy_from_config_path(method_config_path)

    model = M8V1CRuntimeDetectionModel(
        cfg=model_path.as_posix(),
        ch=channels,
        nc=nc,
        verbose=False,
        runtime_policy=policy,
    )
    model.eval()
    fake_batch = {
        "img": torch.rand(2, channels, 640, 640),
        "batch_idx": torch.tensor([0, 1], dtype=torch.int64),
        "bboxes": torch.tensor(
            [
                [0.50, 0.50, 0.03, 0.03],
                [0.50, 0.50, 0.20, 0.20],
            ],
            dtype=torch.float32,
        ),
    }
    model.set_runtime_batch_metadata(fake_batch)
    with torch.no_grad():
        _ = model(fake_batch["img"])
    runtime_ok = bool(model.last_runtime_application.get("applied")) and bool(model.runtime_method_integration_verified)
    return runtime_ok, json.dumps(model.last_runtime_application, ensure_ascii=True)


def build_command_draft(
    args: argparse.Namespace,
    resolved_data_path: Path,
    resolved_model_path: Path,
    resolved_method_config_path: Path,
) -> str:
    common_lines = [
        ".venv/bin/python scripts/train_runtime_weighted_yolo.py \\",
        f"  --data {resolved_data_path.as_posix()} \\",
        f"  --model {resolved_model_path.as_posix()} \\",
        f"  --method-config {resolved_method_config_path.as_posix()} \\",
        f"  --epochs {args.epochs} \\",
        f"  --imgsz {args.imgsz} \\",
        f"  --batch {args.batch} \\",
        f"  --device {args.device} \\",
        f"  --workers {args.workers} \\",
        f"  --patience {args.patience} \\",
        f"  --project {args.project} \\",
        f"  --name {args.name} \\",
        f"  --seed {args.seed} \\",
    ]
    if args.weights:
        common_lines.append(f"  --weights {args.weights} \\")
    dry_run_lines = common_lines + ["  --dry-run"]
    execute_lines = common_lines + ["  --execute"]
    return "\n".join(
        [
            "# Dry-run/preflight command",
            *dry_run_lines,
            "",
            "# Execute command template",
            "# Only run after human approval in your workflow.",
            *execute_lines,
            "",
        ]
    )


def build_readme(summary: dict[str, Any], artifact_paths: list[str], check_rows: list[CheckRow]) -> str:
    failing_items = [row.item for row in check_rows if row.status != "pass"]
    lines = [
        "# Runtime-Weighted YOLO Training Entrypoint",
        "",
        "This directory contains the repo-local runtime-weighted YOLO preflight bundle.",
        "",
        "## Summary",
        f"- Phase: `{summary['phase']}`",
        f"- Status: `{summary['status']}`",
        f"- Dry-run passed: `{summary['dry_run_passed']}`",
        f"- Runtime method integration verified: `{summary['runtime_method_integration_verified']}`",
        f"- Training execution ready: `{summary['training_execution_ready']}`",
        f"- Next allowed step: `{summary['next_allowed_step']}`",
        "",
        "## Guardrails",
        "- No training was executed in this preparation step unless `--execute` is explicitly used.",
        "- No evaluation, inference, or prediction was executed in dry-run mode.",
        "- No dataset files or checkpoints were mutated in dry-run mode.",
        "",
        "## Generated Files",
    ]
    lines.extend(f"- `{path}`" for path in artifact_paths)
    lines.extend(["", "## Blocking Findings"])
    if failing_items:
        lines.extend(f"- {item}" for item in failing_items)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def emit_summary(summary: dict[str, Any], artifact_paths: list[str]) -> None:
    for key in SUMMARY_FIELDS:
        print(f"{key}={summary[key]}")
    print("generated_artifact_paths=")
    for artifact_path in artifact_paths:
        print(artifact_path)


def main() -> None:
    args = build_parser().parse_args()
    if args.execute and args.dry_run:
        raise SystemExit("Pass either --dry-run or --execute, not both.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resolved_data_path, data_candidates = resolve_existing_path(args.data)
    resolved_model_path, model_candidates = resolve_existing_path(args.model, search_dirs=(Path("config"), Path("configs")))
    resolved_method_config_path, method_candidates = resolve_existing_path(args.method_config, search_dirs=(Path("configs/method"),))

    dataset_config_exists = resolved_data_path.exists()
    model_config_exists = resolved_model_path.exists()
    method_config_exists = resolved_method_config_path.exists()

    check_rows: list[CheckRow] = [
        CheckRow(
            item="dataset config exists",
            status="pass" if dataset_config_exists else "fail",
            evidence=", ".join(candidate.as_posix() for candidate in data_candidates),
            notes=resolved_data_path.as_posix() if dataset_config_exists else "dataset YAML not found",
        ),
        CheckRow(
            item="model config exists",
            status="pass" if model_config_exists else "fail",
            evidence=", ".join(candidate.as_posix() for candidate in model_candidates),
            notes=resolved_model_path.as_posix() if model_config_exists else "model YAML not found",
        ),
        CheckRow(
            item="method config exists",
            status="pass" if method_config_exists else "fail",
            evidence=", ".join(candidate.as_posix() for candidate in method_candidates),
            notes=resolved_method_config_path.as_posix() if method_config_exists else "method config YAML not found",
        ),
    ]

    dataset_yaml_valid = False
    dataset_yaml_notes = "dataset config missing"
    data_payload: dict[str, Any] = {}
    if dataset_config_exists:
        dataset_yaml_valid, dataset_yaml_notes, data_payload = validate_dataset_config(resolved_data_path)
    check_rows.append(
        CheckRow(
            item="dataset config has YOLO-required fields",
            status="pass" if dataset_yaml_valid else "fail",
            evidence=resolved_data_path.as_posix() if dataset_config_exists else "dataset config missing",
            notes=dataset_yaml_notes,
        )
    )

    method_config_policy_verified = False
    method_policy_notes = "method config missing"
    if method_config_exists:
        method_config_policy_verified, method_policy_notes = validate_method_policy(resolved_method_config_path)
    check_rows.append(
        CheckRow(
            item="method config represents runtime-usable M8_v1c policy",
            status="pass" if method_config_policy_verified else "fail",
            evidence=resolved_method_config_path.as_posix() if method_config_exists else "method config missing",
            notes=method_policy_notes,
        )
    )

    runtime_method_integration_verified = False
    runtime_notes = "runtime verification skipped"
    if dataset_yaml_valid and model_config_exists and method_config_policy_verified:
        runtime_method_integration_verified, runtime_notes = verify_runtime_integration(
            resolved_model_path,
            data_payload,
            resolved_method_config_path,
        )
    check_rows.append(
        CheckRow(
            item="runtime method integration verified",
            status="pass" if runtime_method_integration_verified else "fail",
            evidence=f"{resolved_model_path.as_posix()}; {resolved_method_config_path.as_posix()}",
            notes=runtime_notes,
        )
    )

    dry_run_mode = not args.execute
    dry_run_passed = (
        dataset_config_exists
        and model_config_exists
        and method_config_exists
        and dataset_yaml_valid
        and method_config_policy_verified
        and runtime_method_integration_verified
    )
    training_execution_ready = dry_run_passed
    training_allowed = bool(args.execute and training_execution_ready)
    training_executed = False

    status = "m8_v1c_training_entrypoint_prepared" if dry_run_passed else "m8_v1c_training_entrypoint_blocked"
    next_allowed_step = (
        "review_m8_v1c_training_command_and_execute_when_ready"
        if training_execution_ready
        else "fix_m8_v1c_runtime_or_policy_before_training"
    )

    summary = {
        "phase": "M8_v1c.training_entrypoint",
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "resolved_dataset_config": resolved_data_path.as_posix(),
        "resolved_model_config": resolved_model_path.as_posix(),
        "resolved_method_config": resolved_method_config_path.as_posix(),
        "dataset_config_exists": dataset_config_exists,
        "model_config_exists": model_config_exists,
        "method_config_exists": method_config_exists,
        "method_config_policy_verified": method_config_policy_verified,
        "runtime_method_integration_verified": runtime_method_integration_verified,
        "dry_run_passed": dry_run_passed,
        "execute_flag_required": True,
        "training_execution_ready": training_execution_ready,
        "training_allowed": training_allowed,
        "training_executed": training_executed,
        "evaluation_executed": False,
        "inference_executed": False,
        "prediction_executed": False,
        "dataset_mutated": False,
        "checkpoint_loaded": False,
        "checkpoint_mutated": False,
        "next_allowed_step": next_allowed_step,
        "dry_run_mode": dry_run_mode,
    }

    artifact_paths = [
        SUMMARY_PATH.as_posix(),
        CHECK_PATH.as_posix(),
        COMMAND_DRAFT_PATH.as_posix(),
        MANIFEST_PATH.as_posix(),
        README_PATH.as_posix(),
    ]

    if args.execute:
        if not training_execution_ready:
            write_json(SUMMARY_PATH, summary)
            write_csv(CHECK_PATH, check_rows)
            COMMAND_DRAFT_PATH.write_text(
                build_command_draft(args, resolved_data_path, resolved_model_path, resolved_method_config_path),
                encoding="utf-8",
            )
            write_json(
                MANIFEST_PATH,
                {
                    "phase": "M8_v1c.training_entrypoint",
                    "status": status,
                    "training_executed": False,
                    "evaluation_executed": False,
                    "inference_executed": False,
                    "prediction_executed": False,
                    "dataset_mutated": False,
                    "checkpoint_loaded": False,
                    "checkpoint_mutated": False,
                    "training_allowed": training_allowed,
                    "execute_flag_required": True,
                    "runtime_method_integration_verified": runtime_method_integration_verified,
                    "dry_run_mode": dry_run_mode,
                    "generated_at_utc": summary["generated_at_utc"],
                    "next_allowed_step": next_allowed_step,
                },
            )
            README_PATH.write_text(build_readme(summary, artifact_paths, check_rows), encoding="utf-8")
            emit_summary(summary, artifact_paths)
            raise SystemExit("Execution refused: runtime integration or method-config validation is incomplete.")

        from ultralytics import YOLO

        model = YOLO(resolved_model_path.as_posix())
        train_kwargs = {
            "trainer": M8V1CPolicyMiningTrainer,
            "data": resolved_data_path.as_posix(),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "project": args.project,
            "name": args.name,
            "seed": args.seed,
            "method_config": resolved_method_config_path.as_posix(),
        }
        if args.weights:
            train_kwargs["pretrained"] = args.weights
        model.train(**train_kwargs)
        training_executed = True

        summary["training_executed"] = True
        summary["training_allowed"] = True
        summary["checkpoint_loaded"] = bool(args.weights)
        summary["status"] = "m8_v1c_training_entrypoint_executed"
        summary["next_allowed_step"] = "review_m8_v1c_training_outputs"
        write_json(SUMMARY_PATH, summary)
        write_csv(CHECK_PATH, check_rows)
        COMMAND_DRAFT_PATH.write_text(
            build_command_draft(args, resolved_data_path, resolved_model_path, resolved_method_config_path),
            encoding="utf-8",
        )
        write_json(
            MANIFEST_PATH,
            {
                "phase": "M8_v1c.training_entrypoint",
                "status": summary["status"],
                "training_executed": True,
                "evaluation_executed": False,
                "inference_executed": False,
                "prediction_executed": False,
                "dataset_mutated": False,
                "checkpoint_loaded": bool(args.weights),
                "checkpoint_mutated": False,
                "training_allowed": True,
                "execute_flag_required": True,
                "runtime_method_integration_verified": runtime_method_integration_verified,
                "dry_run_mode": False,
                "generated_at_utc": summary["generated_at_utc"],
                "next_allowed_step": summary["next_allowed_step"],
            },
        )
        README_PATH.write_text(build_readme(summary, artifact_paths, check_rows), encoding="utf-8")
    else:
        write_json(SUMMARY_PATH, summary)
        write_csv(CHECK_PATH, check_rows)
        COMMAND_DRAFT_PATH.write_text(
            build_command_draft(args, resolved_data_path, resolved_model_path, resolved_method_config_path),
            encoding="utf-8",
        )
        write_json(
            MANIFEST_PATH,
            {
                "phase": "M8_v1c.training_entrypoint",
                "status": status,
                "training_executed": False,
                "evaluation_executed": False,
                "inference_executed": False,
                "prediction_executed": False,
                "dataset_mutated": False,
                "checkpoint_loaded": False,
                "checkpoint_mutated": False,
                "training_allowed": training_allowed,
                "execute_flag_required": True,
                "runtime_method_integration_verified": runtime_method_integration_verified,
                "dry_run_mode": dry_run_mode,
                "generated_at_utc": summary["generated_at_utc"],
                "next_allowed_step": next_allowed_step,
            },
        )
        README_PATH.write_text(build_readme(summary, artifact_paths, check_rows), encoding="utf-8")

    emit_summary(summary, artifact_paths)


if __name__ == "__main__":
    main()
