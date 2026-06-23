"""Thin Ultralytics YOLO wrapper for baseline training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from xai_evidence_sod.utils.config import dump_yaml_config, ensure_dir, ensure_file


class YoloBaselineRunner:
    """Small helper around Ultralytics YOLO to keep scripts focused."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = YOLO(model_name)

    def train(self, train_args: dict[str, Any]) -> Any:
        """Run baseline training with Ultralytics."""

        return self.model.train(**train_args)

    def validate(self, val_args: dict[str, Any]) -> Any:
        """Run model validation."""

        return self.model.val(**val_args)

    def predict(self, predict_args: dict[str, Any]) -> list[Any]:
        """Run predictions for qualitative inspection."""

        return self.model.predict(**predict_args)

    @staticmethod
    def prepare_run_dir(project_dir: str | Path, run_name: str) -> Path:
        """Create and return a run directory."""

        return ensure_dir(Path(project_dir) / run_name, "run directory", create=True)

    @staticmethod
    def write_run_config(config: dict[str, Any], output_path: str | Path) -> Path:
        """Persist the effective config for reproducibility."""

        cleaned = {key: value for key, value in config.items() if key != "_config_path"}
        return dump_yaml_config(cleaned, output_path)

    @staticmethod
    def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Persist a JSON artifact."""

        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    @staticmethod
    def require_weights(weights_path: str | Path) -> Path:
        """Validate a checkpoint path."""

        return ensure_file(weights_path, "Model weights")
