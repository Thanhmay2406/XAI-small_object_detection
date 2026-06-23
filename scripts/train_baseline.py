"""Train a baseline Ultralytics YOLO detector for the drill-bit dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_evidence_sod.models import YoloBaselineRunner
from xai_evidence_sod.utils.config import ensure_file, load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline YOLO detector.")
    parser.add_argument("--config", default="configs/train/baseline_drill_bit_yolov8n.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None, help="Optional dataset fraction for smoke tests.")
    parser.add_argument("--model", default=None, help="Optional model override such as yolov8n.pt or yolov8n.yaml")
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--name", default=None, help="Optional run name override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config)
    dataset_yaml = ensure_file(config["dataset"], "Dataset YAML")

    run_name = args.name or str(config["experiment_name"])
    project_dir = args.project_dir or str(config["project_dir"])
    run_dir = YoloBaselineRunner.prepare_run_dir(project_dir, run_name)

    effective_model = args.model or str(config["model"])
    runner = YoloBaselineRunner(effective_model)
    YoloBaselineRunner.write_run_config(config, run_dir / "config.yaml")

    train_args = {
        "data": str(dataset_yaml),
        "imgsz": int(args.imgsz or config["imgsz"]),
        "epochs": int(args.epochs or config["epochs"]),
        "batch": int(args.batch or config["batch"]),
        "device": args.device or config["device"],
        "project": str(Path(project_dir).resolve()),
        "name": run_name,
        "seed": int(args.seed if args.seed is not None else config["seed"]),
        "workers": int(config.get("workers", 4)),
        "patience": int(config.get("patience", 20)),
        "exist_ok": bool(config.get("exist_ok", True)),
        "pretrained": bool(config.get("pretrained", True)),
    }
    if args.fraction is not None:
        train_args["fraction"] = float(args.fraction)

    results = runner.train(train_args)

    metrics_payload = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(Path(results.save_dir) / "weights" / "best.pt"),
        "last_checkpoint": str(Path(results.save_dir) / "weights" / "last.pt"),
        "train_args": train_args,
    }
    YoloBaselineRunner.write_json(metrics_payload, run_dir / "train_run.json")

    summary_lines = [
        "# Phase 3 Baseline Training Summary",
        "",
        f"- Config: `{Path(args.config).resolve()}`",
        f"- Dataset: `{dataset_yaml}`",
        f"- Model: `{effective_model}`",
        f"- Run directory: `{Path(results.save_dir).resolve()}`",
        f"- Best checkpoint: `{Path(results.save_dir).resolve() / 'weights' / 'best.pt'}`",
        f"- Last checkpoint: `{Path(results.save_dir).resolve() / 'weights' / 'last.pt'}`",
        "",
        "Use the evaluation script next to generate baseline metrics and artifact summaries.",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
