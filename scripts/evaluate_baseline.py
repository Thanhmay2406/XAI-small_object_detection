"""Evaluate a trained baseline YOLO model and export baseline artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xai_evidence_sod.evaluation import (
    analyze_class_error_cases,
    collect_prediction_samples,
    dump_json,
    summarize_validation_results,
    write_markdown_report,
    write_prediction_csv,
)
from xai_evidence_sod.models import YoloBaselineRunner
from xai_evidence_sod.utils.config import ensure_dir, ensure_file, load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the baseline YOLO detector.")
    parser.add_argument("--config", default="configs/train/baseline_drill_bit_yolov8n.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default=None, help="Dataset split to evaluate, default from config.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--output", default=None, help="Evaluation output directory.")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config)
    dataset_yaml = ensure_file(config["dataset"], "Dataset YAML")
    weights_path = YoloBaselineRunner.require_weights(args.weights)
    output_dir = ensure_dir(args.output or config["eval_project_dir"], "Evaluation output directory", create=True)

    runner = YoloBaselineRunner(str(weights_path))
    split = args.split or str(config.get("split", "test"))
    imgsz = int(args.imgsz or config["imgsz"])
    conf = float(args.conf if args.conf is not None else config.get("conf", 0.25))
    iou = float(args.iou if args.iou is not None else config.get("iou", 0.5))

    val_results = runner.validate(
        {
            "data": str(dataset_yaml),
            "split": split,
            "imgsz": imgsz,
            "device": args.device or config["device"],
            "project": str(output_dir.resolve()),
            "name": "val",
            "plots": bool(config.get("save_plots", True)),
            "save_json": bool(config.get("save_json", True)),
            "exist_ok": True,
        }
    )

    split_dir = _resolve_split_dir(dataset_yaml, split)
    prediction_results = runner.predict(
        {
            "source": str(split_dir),
            "imgsz": imgsz,
            "device": args.device or config["device"],
            "project": str(output_dir.resolve()),
            "name": "prediction_samples",
            "save": True,
            "save_txt": False,
            "save_conf": True,
            "conf": conf,
            "iou": iou,
            "exist_ok": True,
            "verbose": False,
        }
    )

    class_names = list(val_results.names.values()) if isinstance(val_results.names, dict) else list(val_results.names)
    metrics_summary = summarize_validation_results(val_results, class_names)
    prediction_rows = collect_prediction_samples(prediction_results)

    metrics_json = dump_json(metrics_summary, output_dir / "metrics_overall.json")
    write_prediction_csv(prediction_rows, output_dir / "prediction_rows.csv")
    error_cases_csv = analyze_class_error_cases(
        dataset_yaml_path=dataset_yaml,
        prediction_results=prediction_results,
        target_class_name=str(config.get("target_error_class", "Chipped")),
        output_csv_path=output_dir / "chipped_error_cases.csv",
        iou_threshold=iou,
    )
    report_md = write_markdown_report(
        metrics_summary=metrics_summary,
        target_class_name=str(config.get("target_error_class", "Chipped")),
        error_case_csv=error_cases_csv,
        output_path=output_dir / "baseline_eval_report.md",
    )

    summary = {
        "metrics_json": str(metrics_json),
        "report_md": str(report_md),
        "prediction_dir": str((output_dir / "prediction_samples").resolve()),
        "error_cases_csv": str(error_cases_csv),
        "validation_plot_dir": str(Path(val_results.save_dir).resolve()),
    }
    dump_json(summary, output_dir / "artifact_index.json")


def _resolve_split_dir(dataset_yaml: str | Path, split: str) -> Path:
    import yaml

    dataset_yaml = Path(dataset_yaml).expanduser().resolve()
    with dataset_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if data.get("path"):
        raw_root = Path(str(data["path"]))
        if raw_root.is_absolute():
            dataset_root = raw_root.resolve()
        else:
            candidate_roots = [
                (dataset_yaml.parent / raw_root).resolve(),
                (Path.cwd() / raw_root).resolve(),
            ]
            dataset_root = next((candidate for candidate in candidate_roots if candidate.exists()), candidate_roots[-1])
    else:
        dataset_root = dataset_yaml.parent.resolve()
    split_path = Path(str(data[split]))
    if split_path.is_absolute():
        return split_path
    candidates = [
        (dataset_root / split_path).resolve(),
        (dataset_yaml.parent / split_path).resolve(),
        (Path.cwd() / split_path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


if __name__ == "__main__":
    main()
