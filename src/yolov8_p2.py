import argparse
import os
from pathlib import Path
from typing import Any

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model with the P2 feature pyramid head enabled."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset YAML.")
    parser.add_argument(
        "--model",
        required=True,
        help="Ultralytics model config or checkpoint. Use a *-p2.yaml config to keep the P2 head.",
    )
    parser.add_argument(
        "--weights",
        help="Optional pretrained weights. Use a local .pt path to stay offline-safe.",
    )
    parser.add_argument("--epochs", type=int, required=True, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, required=True, help="Input image size.")
    parser.add_argument("--batch", type=int, required=True, help="Batch size.")
    parser.add_argument("--device", required=True, help="Training device, for example 0 or cpu.")
    parser.add_argument("--workers", type=int, required=True, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, required=True, help="Early stopping patience.")
    parser.add_argument("--project", type=Path, required=True, help="Output project dir.")
    parser.add_argument("--name", required=True, help="Run name inside the project dir.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine learning-rate schedule.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run dir.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the resolved training arguments before launching.",
    )
    return parser


def resolve_model(model_ref: str, weights_ref: str | None):
    from ultralytics import YOLO

    model = YOLO(model_ref)
    if weights_ref:
        model = model.load(weights_ref)
    return model


def build_train_args(args: argparse.Namespace) -> dict[str, Any]:
    train_args: dict[str, Any] = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(args.project.resolve()),
        "name": args.name,
        "seed": args.seed,
        "exist_ok": args.exist_ok,
        "cache": args.cache,
        "cos_lr": args.cos_lr,
    }
    train_args["device"] = args.device
    return train_args


def ensure_runtime_dirs() -> None:
    # Avoid Matplotlib cache warnings in restricted environments.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ensure_runtime_dirs()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    model = resolve_model(args.model, args.weights.strip() if args.weights else None)
    train_args = build_train_args(args)

    if args.verbose:
        print("Resolved YOLOv8 P2 training arguments:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")

    model.train(**train_args)


if __name__ == "__main__":
    main()
