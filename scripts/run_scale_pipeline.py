"""Guarded all-in-one scale-policy pipeline.

This entrypoint keeps the default behavior non-executing. It reuses the
existing analysis/runtime integration pieces in this repo, adds coefficient
sensitivity calibration, exports an instance-aware runtime policy, and only
trains when both `--stage train` and `--execute` are present.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.m8_v1c_runtime import runtime_policy_from_config_path
from scripts.analyze_scale_policy import run_analysis as run_metric_analysis
from scripts.export_runtime_policy import export_runtime_payload, load_yaml_mapping

STATUS_MISSING_EXECUTE = "m8_v1c_blocked_missing_execute_flag"
STATUS_INVALID_DATASET = "m8_v1c_blocked_invalid_dataset_yaml"
STATUS_MISSING_CHECKPOINT = "m8_v1c_blocked_missing_checkpoint"
STATUS_INVALID_RUNTIME_POLICY = "m8_v1c_blocked_invalid_runtime_policy"
STATUS_UNRESOLVED_FEATURE_SCALES = "m8_v1c_blocked_unresolved_feature_scales"
STATUS_ANALYSIS_COMPLETED = "m8_v1c_analysis_completed_no_training"
STATUS_DRY_RUN_PASSED = "m8_v1c_dry_run_passed_no_training"
STATUS_TRAINING_EXECUTED = "m8_v1c_training_executed"

ANALYSIS_DIRNAME = "analysis"
TRAIN_PROJECT_DIRNAME = "training_runs"
DEFAULT_MODEL_YAML = Path("configs/yolov8s-p2.yaml")


@dataclass
class GuardrailRow:
    item: str
    status: str
    evidence: str
    notes: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded all-in-one scale-policy pipeline.")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the baseline checkpoint.")
    parser.add_argument(
        "--model-yaml",
        type=Path,
        default=None,
        help="Optional model YAML. Defaults to configs/yolov8s-p2.yaml when present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/scale_pipeline"),
        help="Artifact directory for the full pipeline.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for analysis and training.")
    parser.add_argument("--max-images", type=int, default=64, help="Maximum number of images for analysis.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size for train stage.")
    parser.add_argument("--device", default="cpu", help="Torch device for analysis/train.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Analysis split.")
    parser.add_argument(
        "--stage",
        choices=["analysis", "calibrate", "export", "dry-run", "train"],
        required=True,
        help="Pipeline stage to run.",
    )
    parser.add_argument("--execute", action="store_true", help="Required together with --stage train to execute training.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for train stage.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers for train stage.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience for train stage.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for train stage.")
    parser.add_argument("--run-name", default="m8_v1c_instance_aware", help="Run name for train stage.")
    parser.add_argument(
        "--min-objects-for-policy",
        type=int,
        default=25,
        help="Minimum objects required before auto-proposing a non-identity policy.",
    )
    parser.add_argument(
        "--min-best-minus-second-margin",
        type=float,
        default=0.05,
        help="Minimum evidence margin before auto-proposing a non-identity policy.",
    )
    return parser


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_guardrail_csv(path: Path, rows: list[GuardrailRow]) -> None:
    write_csv(path, [asdict(row) for row in rows], ["item", "status", "evidence", "notes"])


def resolve_model_yaml(model_yaml: Path | None) -> Path | None:
    if model_yaml is not None:
        return model_yaml.resolve()
    if DEFAULT_MODEL_YAML.exists():
        return DEFAULT_MODEL_YAML.resolve()
    return None


def validate_dataset_yaml(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"dataset yaml not found: {path}"
    payload = load_yaml_mapping(path)
    has_split = any(key in payload for key in ("train", "val", "test"))
    has_names = "names" in payload
    if not has_split or not has_names:
        return False, f"missing required fields: split_present={has_split}, names_present={has_names}"
    return True, "dataset yaml shape validated"


def stage_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "analysis_dir": output_dir / ANALYSIS_DIRNAME,
        "object_scale_scores": output_dir / "object_scale_scores.csv",
        "scale_evidence_metrics": output_dir / "scale_evidence_metrics.csv",
        "coefficient_sensitivity": output_dir / "coefficient_sensitivity.csv",
        "coefficient_calibration_summary": output_dir / "coefficient_calibration_summary.json",
        "policy_candidate": output_dir / "policy_candidate.yaml",
        "runtime_policy": output_dir / "runtime_policy.yaml",
        "spatial_runtime_policy_preview": output_dir / "spatial_runtime_policy_preview.json",
        "runtime_validation_summary": output_dir / "runtime_validation_summary.json",
        "non_execution_manifest": output_dir / "non_execution_manifest.json",
        "execution_manifest": output_dir / "execution_manifest.json",
        "method_summary": output_dir / "method_summary.json",
        "method_report": output_dir / "method_report.md",
        "readme": output_dir / "README.md",
        "guardrail_check": output_dir / "guardrail_check.csv",
    }


def run_analysis_stage(args: argparse.Namespace, output_dir: Path, resolved_model_yaml: Path) -> dict[str, Any]:
    analysis_dir = output_dir / ANALYSIS_DIRNAME
    analysis_args = argparse.Namespace(
        model=args.checkpoint,
        model_config=resolved_model_yaml,
        data=args.dataset_yaml,
        split=args.split,
        imgsz=args.imgsz,
        max_images=args.max_images,
        device=args.device,
        output_dir=analysis_dir,
        min_objects_for_policy=args.min_objects_for_policy,
        min_best_minus_second_margin=args.min_best_minus_second_margin,
        dry_run=False,
    )
    run_metric_analysis(analysis_args)

    aliases = {
        analysis_dir / "m8_v1c_xai_metric_object_scores.csv": output_dir / "object_scale_scores.csv",
        analysis_dir / "m8_v1c_scale_evidence_metrics.csv": output_dir / "scale_evidence_metrics.csv",
        analysis_dir / "m8_v1c_metric_policy_candidate.yaml": output_dir / "policy_candidate.yaml",
    }
    for source, destination in aliases.items():
        shutil.copyfile(source, destination)

    summary_path = analysis_dir / "m8_v1c_metric_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def compute_sensitivity_profiles(object_csv_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles = {
        "baseline": (0.5, 0.3, 0.2),
        "energy_heavy": (0.7, 0.2, 0.1),
        "peak_heavy": (0.4, 0.5, 0.1),
        "leakage_guarded": (0.45, 0.2, 0.35),
    }
    grouped: dict[tuple[str, str, str], list[float]] = {}
    with object_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            size_group = str(row["object_size_group"])
            scale = str(row["scale"])
            if size_group == "tiny":
                continue
            energy = float(row["energy_in_box"])
            leakage = float(row["background_leakage"])
            peak = float(row["peak_alignment_score"])
            for profile_name, (w_energy, w_peak, w_guard) in profiles.items():
                score = (w_energy * energy) + (w_peak * peak) + (w_guard * (1.0 - leakage))
                grouped.setdefault((profile_name, size_group, scale), []).append(score)

    rows: list[dict[str, Any]] = []
    stability: dict[str, dict[str, str]] = {profile_name: {} for profile_name in profiles}
    for profile_name in profiles:
        for size_group in ("small", "medium", "large"):
            scale_rows: list[tuple[str, float, int]] = []
            for scale_name in ("P2", "P3", "P4", "P5"):
                scores = grouped.get((profile_name, size_group, scale_name), [])
                if not scores:
                    continue
                mean_score = sum(scores) / len(scores)
                scale_rows.append((scale_name, mean_score, len(scores)))
            selected_scale = max(scale_rows, key=lambda item: (item[1], item[0]))[0] if scale_rows else "identity"
            stability[profile_name][size_group] = selected_scale
            for scale_name, mean_score, object_count in scale_rows:
                rows.append(
                    {
                        "profile": profile_name,
                        "size_group": size_group,
                        "scale": scale_name,
                        "object_count": object_count,
                        "mean_score": mean_score,
                        "selected_scale": selected_scale,
                    }
                )

    baseline = stability["baseline"]
    consistent_profiles = 0
    for profile_name, selections in stability.items():
        if selections == baseline:
            consistent_profiles += 1
    summary = {
        "profiles": {name: {"weights": list(weights), "selected_scales": stability[name]} for name, weights in profiles.items()},
        "baseline_profile": "baseline",
        "profiles_matching_baseline": consistent_profiles,
        "policy_pattern_stable": consistent_profiles == len(profiles),
    }
    return rows, summary


def run_calibration_stage(output_dir: Path) -> dict[str, Any]:
    sensitivity_rows, sensitivity_summary = compute_sensitivity_profiles(output_dir / "object_scale_scores.csv")
    write_csv(
        output_dir / "coefficient_sensitivity.csv",
        sensitivity_rows,
        ["profile", "size_group", "scale", "object_count", "mean_score", "selected_scale"],
    )
    write_json(output_dir / "coefficient_calibration_summary.json", sensitivity_summary)
    return sensitivity_summary


def run_export_stage(output_dir: Path) -> dict[str, Any]:
    candidate_path = output_dir / "policy_candidate.yaml"
    candidate = load_yaml_mapping(candidate_path)
    runtime_payload = export_runtime_payload(
        candidate,
        policy_candidate_path=candidate_path,
        small_max_pixel_area=1024.0,
        medium_max_pixel_area=9216.0,
    )
    runtime_policy_path = output_dir / "runtime_policy.yaml"
    runtime_policy_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_policy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime_payload, handle, sort_keys=False)

    policy = runtime_policy_from_config_path(runtime_policy_path)
    preview = {
        "method_name": policy.method_name,
        "runtime_application_mode": policy.application_mode,
        "supported_scales": list(policy.supported_scales),
        "size_thresholds": {
            "small_max_pixel_area": policy.size_thresholds.small_max_area,
            "medium_max_pixel_area": policy.size_thresholds.medium_max_area,
        },
        "size_groups": {
            group_name: {
                "mode": group.mode,
                "preferred_scale": group.preferred_scale,
                "weight_delta": group.weight_delta,
            }
            for group_name, group in policy.groups.items()
        },
    }
    write_json(output_dir / "spatial_runtime_policy_preview.json", preview)
    return preview


def run_runtime_validation_stage(
    *,
    dataset_yaml: Path,
    model_yaml: Path,
    runtime_policy_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from models.m8_v1c_runtime import M8V1CRuntimeDetectionModel
    import torch

    dataset_payload = load_yaml_mapping(dataset_yaml)
    channels = int(dataset_payload.get("channels", 3) or 3)
    nc = int(dataset_payload["nc"])
    policy = runtime_policy_from_config_path(runtime_policy_path)

    model = M8V1CRuntimeDetectionModel(
        cfg=str(model_yaml),
        ch=channels,
        nc=nc,
        verbose=False,
        runtime_policy=policy,
    )
    model.eval()
    fake_batch = {
        "img": torch.rand(2, channels, 640, 640),
        "batch_idx": torch.tensor([0, 0, 1], dtype=torch.int64),
        "bboxes": torch.tensor(
            [
                [0.25, 0.25, 0.03, 0.03],
                [0.75, 0.75, 0.20, 0.20],
                [0.50, 0.50, 0.10, 0.10],
            ],
            dtype=torch.float32,
        ),
        "cls": torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
    }
    model.set_runtime_batch_metadata(fake_batch)
    with torch.no_grad():
        _ = model(fake_batch["img"])

    summary = {
        "status": "m8_v1c_runtime_validation_passed" if model.last_runtime_application.get("applied") else STATUS_INVALID_RUNTIME_POLICY,
        "runtime_application_mode": policy.application_mode,
        "runtime_method_integration_verified": bool(model.runtime_method_integration_verified),
        "last_runtime_application": model.last_runtime_application,
    }
    write_json(output_dir / "runtime_validation_summary.json", summary)
    return summary


def train_with_runtime_policy(
    *,
    dataset_yaml: Path,
    model_yaml: Path,
    checkpoint: Path,
    runtime_policy_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from ultralytics import YOLO

    from models.m8_v1c_runtime import M8V1CPolicyMiningTrainer

    project_dir = output_dir / TRAIN_PROJECT_DIRNAME
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_yaml))
    train_kwargs = {
        "trainer": M8V1CPolicyMiningTrainer,
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(project_dir),
        "name": args.run_name,
        "seed": args.seed,
        "method_config": str(runtime_policy_path),
        "pretrained": str(checkpoint),
    }
    results = model.train(**train_kwargs)
    return {
        "status": STATUS_TRAINING_EXECUTED,
        "training_results_type": type(results).__name__,
        "project_dir": str(project_dir.resolve()),
        "run_name": args.run_name,
    }


def build_report(summary: dict[str, Any], guardrail_rows: list[GuardrailRow], generated_paths: list[Path]) -> str:
    lines = [
        "# M8_v1c.2 Instance-Aware All-In-One",
        "",
        "## Summary",
        f"- Stage: `{summary['requested_stage']}`",
        f"- Status: `{summary['status']}`",
        f"- Training executed: `{summary['training_executed']}`",
        f"- Runtime validation status: `{summary.get('runtime_validation_status', 'not_run')}`",
        "",
        "## Guardrails",
    ]
    for row in guardrail_rows:
        lines.append(f"- {row.item}: {row.status} ({row.notes})")
    lines.extend(["", "## Generated Files"])
    for path in generated_paths:
        if path.exists():
            lines.append(f"- `{path.name}`")
    lines.append("")
    return "\n".join(lines)


def build_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# M8_v1c.2 Instance-Aware Artifacts",
            "",
            "This directory was generated by `scripts/run_scale_pipeline.py`.",
            "",
            f"- requested_stage: `{summary['requested_stage']}`",
            f"- status: `{summary['status']}`",
            f"- training_executed: `{summary['training_executed']}`",
            f"- runtime_application_mode: `{summary.get('runtime_application_mode', 'not_exported')}`",
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = stage_paths(output_dir)

    guardrail_rows: list[GuardrailRow] = []
    dataset_ok, dataset_notes = validate_dataset_yaml(args.dataset_yaml)
    guardrail_rows.append(
        GuardrailRow(
            item="dataset yaml validation",
            status="pass" if dataset_ok else "fail",
            evidence=str(args.dataset_yaml.resolve(strict=False)),
            notes=dataset_notes,
        )
    )

    checkpoint_exists = args.checkpoint.exists()
    guardrail_rows.append(
        GuardrailRow(
            item="checkpoint exists",
            status="pass" if checkpoint_exists else "fail",
            evidence=str(args.checkpoint.resolve(strict=False)),
            notes="baseline checkpoint found" if checkpoint_exists else "checkpoint missing",
        )
    )

    resolved_model_yaml = resolve_model_yaml(args.model_yaml)
    model_yaml_ok = resolved_model_yaml is not None and resolved_model_yaml.exists()
    guardrail_rows.append(
        GuardrailRow(
            item="model yaml resolved",
            status="pass" if model_yaml_ok else "fail",
            evidence=str(resolved_model_yaml) if resolved_model_yaml is not None else "None",
            notes="model yaml available" if model_yaml_ok else "could not resolve model yaml for feature scales",
        )
    )

    if not dataset_ok:
        status = STATUS_INVALID_DATASET
    elif not checkpoint_exists:
        status = STATUS_MISSING_CHECKPOINT
    elif not model_yaml_ok:
        status = STATUS_UNRESOLVED_FEATURE_SCALES
    elif args.stage == "train" and not args.execute:
        status = STATUS_MISSING_EXECUTE
    else:
        status = ""

    analysis_summary: dict[str, Any] | None = None
    calibration_summary: dict[str, Any] | None = None
    runtime_preview: dict[str, Any] | None = None
    runtime_validation_summary: dict[str, Any] | None = None
    execution_details: dict[str, Any] | None = None

    if not status:
        analysis_summary = run_analysis_stage(args, output_dir, resolved_model_yaml)
        if args.stage in {"calibrate", "export", "dry-run", "train"}:
            calibration_summary = run_calibration_stage(output_dir)
        if args.stage in {"export", "dry-run", "train"}:
            runtime_preview = run_export_stage(output_dir)
            try:
                runtime_validation_summary = run_runtime_validation_stage(
                    dataset_yaml=args.dataset_yaml.resolve(),
                    model_yaml=resolved_model_yaml,
                    runtime_policy_path=paths["runtime_policy"],
                    output_dir=output_dir,
                )
                if runtime_validation_summary["status"] != "m8_v1c_runtime_validation_passed":
                    status = STATUS_INVALID_RUNTIME_POLICY
            except Exception as exc:
                status = STATUS_INVALID_RUNTIME_POLICY
                runtime_validation_summary = {
                    "status": STATUS_INVALID_RUNTIME_POLICY,
                    "error": str(exc),
                }
                write_json(paths["runtime_validation_summary"], runtime_validation_summary)
        if not status:
            if args.stage == "train":
                execution_details = train_with_runtime_policy(
                    dataset_yaml=args.dataset_yaml.resolve(),
                    model_yaml=resolved_model_yaml,
                    checkpoint=args.checkpoint.resolve(),
                    runtime_policy_path=paths["runtime_policy"],
                    output_dir=output_dir,
                    args=args,
                )
                status = STATUS_TRAINING_EXECUTED
            elif args.stage == "dry-run":
                status = STATUS_DRY_RUN_PASSED
            else:
                status = STATUS_ANALYSIS_COMPLETED

    summary = {
        "phase": "M8_v1c.2",
        "requested_stage": args.stage,
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_yaml": str(args.dataset_yaml.resolve(strict=False)),
        "checkpoint": str(args.checkpoint.resolve(strict=False)),
        "model_yaml": str(resolved_model_yaml) if resolved_model_yaml is not None else None,
        "output_dir": str(output_dir),
        "training_executed": status == STATUS_TRAINING_EXECUTED,
        "runtime_application_mode": runtime_preview["runtime_application_mode"] if runtime_preview else None,
        "runtime_validation_status": runtime_validation_summary["status"] if runtime_validation_summary else "not_run",
        "analysis_summary_path": str((paths["analysis_dir"] / "m8_v1c_metric_summary.json").resolve(strict=False))
        if analysis_summary
        else None,
        "coefficient_calibration_summary_path": str(paths["coefficient_calibration_summary"].resolve(strict=False))
        if calibration_summary
        else None,
        "runtime_policy_path": str(paths["runtime_policy"].resolve(strict=False)) if runtime_preview else None,
        "execution_details": execution_details,
        "guardrail_rows": [asdict(row) for row in guardrail_rows],
    }

    generated_paths = [path for path in paths.values() if path.exists()]
    write_json(paths["method_summary"], summary)
    write_guardrail_csv(paths["guardrail_check"], guardrail_rows)
    paths["method_report"].write_text(build_report(summary, guardrail_rows, generated_paths), encoding="utf-8")
    paths["readme"].write_text(build_readme(summary), encoding="utf-8")

    if status == STATUS_TRAINING_EXECUTED:
        write_json(
            paths["execution_manifest"],
            {
                "phase": "M8_v1c.2",
                "status": status,
                "training_executed": True,
                "evaluation_executed": False,
                "inference_executed": False,
                "prediction_executed": False,
                "dataset_mutated": False,
                "checkpoint_mutated": False,
            },
        )
    else:
        write_json(
            paths["non_execution_manifest"],
            {
                "phase": "M8_v1c.2",
                "status": status,
                "training_executed": False,
                "evaluation_executed": False,
                "inference_executed": False,
                "prediction_executed": False,
                "dataset_mutated": False,
                "checkpoint_mutated": False,
            },
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
