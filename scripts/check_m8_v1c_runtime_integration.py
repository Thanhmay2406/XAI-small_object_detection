"""Synthetic runtime verification for the repo-local M8_v1c integration hook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.m8_v1c_runtime import M8V1CRuntimeDetectionModel, runtime_policy_from_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic-only M8_v1c runtime integration check.")
    parser.add_argument("--model", type=Path, required=True, help="Path to local YOLO model YAML.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset YAML for nc/ch validation.")
    parser.add_argument("--method-config", type=Path, required=True, help="Path to M8_v1c runtime method config YAML.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m8_v1c_runtime_check/m8_v1c_runtime_check_report.json"),
        help="Path to the synthetic runtime check report.",
    )
    return parser


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    data_payload = load_yaml_mapping(args.data)
    if "nc" not in data_payload:
        raise ValueError(f"Dataset YAML must include nc: {args.data}")
    channels = int(data_payload.get("channels", 3) or 3)
    nc = int(data_payload["nc"])

    policy = runtime_policy_from_config_path(args.method_config)
    model = M8V1CRuntimeDetectionModel(
        cfg=str(args.model),
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
                [0.50, 0.50, 0.03, 0.03],  # small
                [0.50, 0.50, 0.20, 0.20],  # large
            ],
            dtype=torch.float32,
        ),
    }
    model.set_runtime_batch_metadata(fake_batch)

    with torch.no_grad():
        _ = model(fake_batch["img"])

    output_payload = {
        "status": "m8_v1c_runtime_check_passed" if model.last_runtime_application.get("applied") else "m8_v1c_runtime_check_failed",
        "model_yaml": str(args.model),
        "method_config": str(args.method_config),
        "runtime_method_integration_verified": bool(model.runtime_method_integration_verified),
        "last_runtime_application": model.last_runtime_application,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
