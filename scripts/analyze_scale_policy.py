"""Analysis-only XAI metric mining for size-aware scale policy discovery.

This script extends the earlier M8 diagnostic direction by computing richer
per-scale evidence metrics for each ground-truth object:
- energy_in_box
- background_leakage
- peak_alignment_score
- combined evidence_score

It does not train, validate via the Ultralytics API, mutate checkpoints, or
modify dataset artifacts. It only performs forward passes on a loaded model to
capture feature activations for evidence mining.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
from PIL import Image
import torch
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCALE_ORDER = ("P2", "P3", "P4", "P5")
CSV_HEADERS = [
    "image_path",
    "label_path",
    "class_id",
    "class_name",
    "object_index",
    "bbox_x_center_norm",
    "bbox_y_center_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "bbox_area_px",
    "object_size_group",
    "scale",
    "feature_shape",
    "energy_in_box",
    "background_leakage",
    "peak_alignment_score",
    "peak_inside_box",
    "evidence_score",
]


@dataclass
class LabelRecord:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class ActivationCollector:
    def __init__(self, torch_model: torch.nn.Module, scale_to_layer: dict[str, int]) -> None:
        self.torch_model = torch_model
        self.scale_to_layer = scale_to_layer
        self.activations: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _extract_4d_tensor(self, output: Any) -> torch.Tensor | None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            return output.detach().cpu()
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor) and item.ndim == 4:
                    return item.detach().cpu()
        return None

    def clear(self) -> None:
        self.activations.clear()

    def register(self) -> None:
        module_list = getattr(self.torch_model, "model", None)
        if module_list is None:
            raise ValueError("Loaded torch model does not expose a .model module list.")

        for scale_name, layer_index in self.scale_to_layer.items():
            if layer_index < 0 or layer_index >= len(module_list):
                raise ValueError(f"Invalid layer index {layer_index} for {scale_name}.")

            def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any, scale: str = scale_name) -> None:
                tensor = self._extract_4d_tensor(output)
                if tensor is not None:
                    self.activations[scale] = tensor

            self.handles.append(module_list[layer_index].register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analysis-only XAI metric mining for scale policy discovery."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to trained YOLO checkpoint .pt.")
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="YOLO model YAML used to resolve Detect input layers.",
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset YAML.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to analyze.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square resize size used for the forward pass.")
    parser.add_argument("--max-images", type=int, default=64, help="Maximum number of images to analyze.")
    parser.add_argument("--device", default="cpu", help="Torch device such as cpu or cuda:0.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/scale_policy_analysis"),
        help="Directory for CSV, summary, report, and policy outputs.",
    )
    parser.add_argument(
        "--min-objects-for-policy",
        type=int,
        default=25,
        help="Minimum objects required in a size group before auto-proposing a non-identity policy.",
    )
    parser.add_argument(
        "--min-best-minus-second-margin",
        type=float,
        default=0.05,
        help="Minimum evidence-score margin between best and second-best scale for auto policy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, resolve hook layers, and write non-execution artifacts only.",
    )
    return parser


def ensure_runtime_dirs() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def normalize_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    return {}


def resolve_split_dir(data_yaml_path: Path, data_config: dict[str, Any], split: str) -> Path:
    if split not in data_config:
        raise ValueError(f"Split '{split}' not found in dataset YAML: {data_yaml_path}")

    split_value = data_config[split]
    if not isinstance(split_value, str) or not split_value.strip():
        raise ValueError(f"Dataset split '{split}' must be a non-empty string path: {data_yaml_path}")

    split_path = Path(split_value)
    if split_path.is_absolute():
        resolved = split_path
    else:
        base_root = data_config.get("path")
        if isinstance(base_root, str) and base_root.strip():
            base_path = Path(base_root)
            if not base_path.is_absolute():
                base_path = (data_yaml_path.parent / base_path).resolve()
        else:
            base_path = data_yaml_path.parent.resolve()
        resolved = (base_path / split_path).resolve()

    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"Resolved image directory does not exist: {resolved}")
    return resolved


def gather_image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_label_path(image_path: Path) -> Path:
    image_posix = image_path.as_posix()
    if "/images/" in image_posix:
        return Path(image_posix.replace("/images/", "/labels/", 1)).with_suffix(".txt")

    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_image_tensor(image_path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image with cv2: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor.unsqueeze(0).to(device)


def parse_label_file(label_path: Path, class_names: dict[int, str], warnings: list[str]) -> list[LabelRecord]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: list[LabelRecord] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            warnings.append(f"Invalid label line skipped: {label_path}:{line_number} -> expected 5 fields.")
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            warnings.append(f"Invalid label line skipped: {label_path}:{line_number} -> non-numeric values.")
            continue
        if width <= 0 or height <= 0:
            warnings.append(f"Invalid bbox skipped: {label_path}:{line_number} -> non-positive width or height.")
            continue
        if class_id not in class_names:
            warnings.append(f"Class id {class_id} missing from names map: {label_path}:{line_number}.")
        records.append(LabelRecord(class_id, x_center, y_center, width, height))
    return records


def detect_entry_from_model_yaml(model_config: dict[str, Any]) -> list[int]:
    head = model_config.get("head")
    if not isinstance(head, list):
        raise ValueError("Model config does not contain a valid 'head' list.")

    detect_entry: list[Any] | None = None
    for entry in head:
        if isinstance(entry, list) and len(entry) >= 3 and str(entry[2]) == "Detect":
            detect_entry = entry
    if detect_entry is None:
        raise ValueError("No Detect layer found in model config head.")

    layer_from = detect_entry[0]
    if not isinstance(layer_from, list):
        raise ValueError("Detect layer 'from' field must be a list.")
    if len(layer_from) not in {3, 4}:
        raise ValueError("Detect layer 'from' field must contain either 3 or 4 indices.")
    return [int(index) for index in layer_from]


def map_indices_to_scales(indices: list[int]) -> dict[str, int]:
    if len(indices) == 4:
        scales = SCALE_ORDER
    elif len(indices) == 3:
        scales = ("P3", "P4", "P5")
    else:
        raise ValueError("Expected 3 or 4 Detect input indices.")
    return dict(zip(scales, indices, strict=True))


def bbox_area_px(width_norm: float, height_norm: float, imgsz: int) -> float:
    return width_norm * imgsz * height_norm * imgsz


def size_group_for_area(area_px: float) -> str:
    if area_px < 32 * 32:
        return "tiny"
    if area_px < 96 * 96:
        return "small"
    if area_px < 224 * 224:
        return "medium"
    return "large"


def feature_shape_string(activation: torch.Tensor | None) -> str:
    if activation is None:
        return ""
    return "x".join(str(dim) for dim in activation.shape)


def build_bbox_indices(label: LabelRecord, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int((label.x_center - (label.width / 2.0)) * width)
    y1 = int((label.y_center - (label.height / 2.0)) * height)
    x2 = int((label.x_center + (label.width / 2.0)) * width)
    y2 = int((label.y_center + (label.height / 2.0)) * height)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def compute_scale_metrics(activation: torch.Tensor, label: LabelRecord) -> dict[str, Any]:
    if activation.ndim != 4:
        return {
            "energy_in_box": 0.0,
            "background_leakage": 1.0,
            "peak_alignment_score": 0.0,
            "peak_inside_box": False,
            "evidence_score": 0.0,
        }

    feature = activation[0]
    energy = feature.abs().mean(dim=0)
    height, width = int(energy.shape[0]), int(energy.shape[1])
    x1, y1, x2, y2 = build_bbox_indices(label, width, height)
    if x2 <= x1 or y2 <= y1:
        return {
            "energy_in_box": 0.0,
            "background_leakage": 1.0,
            "peak_alignment_score": 0.0,
            "peak_inside_box": False,
            "evidence_score": 0.0,
        }

    mask = torch.zeros((height, width), dtype=energy.dtype)
    mask[y1:y2, x1:x2] = 1.0
    mask_sum = float(mask.sum().item())
    total_energy = float(energy.sum().item()) + 1e-8
    inside_energy = float((energy * mask).sum().item())
    outside_energy = max(0.0, total_energy - inside_energy)

    energy_in_box = inside_energy / total_energy

    outside_mask = 1.0 - mask
    outside_pixels = float(outside_mask.sum().item()) + 1e-8
    inside_mean = inside_energy / max(mask_sum, 1.0)
    outside_mean = float((energy * outside_mask).sum().item()) / outside_pixels
    background_leakage = outside_mean / (inside_mean + outside_mean + 1e-8)

    peak_index = int(torch.argmax(energy).item())
    peak_y = peak_index // width
    peak_x = peak_index % width
    peak_inside_box = x1 <= peak_x < x2 and y1 <= peak_y < y2

    bbox_center_x = ((x1 + x2) / 2.0) - 0.5
    bbox_center_y = ((y1 + y2) / 2.0) - 0.5
    dx = float(peak_x) - bbox_center_x
    dy = float(peak_y) - bbox_center_y
    distance = (dx * dx + dy * dy) ** 0.5
    bbox_diag = max((((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5), 1.0)
    peak_alignment_score = max(0.0, 1.0 - (distance / bbox_diag))

    evidence_score = (0.5 * energy_in_box) + (0.3 * peak_alignment_score) + (0.2 * (1.0 - background_leakage))
    return {
        "energy_in_box": energy_in_box,
        "background_leakage": background_leakage,
        "peak_alignment_score": peak_alignment_score,
        "peak_inside_box": peak_inside_box,
        "evidence_score": evidence_score,
    }


def build_manifest() -> dict[str, Any]:
    return {
        "phase": "M8_v1c.metric_mining",
        "script_purpose": "analysis_only",
        "training_executed": False,
        "evaluation_executed": False,
        "ultralytics_train_api_called": False,
        "ultralytics_val_api_called": False,
        "ultralytics_predict_api_called": False,
        "model_weights_modified": False,
        "dataset_files_modified": False,
        "labels_modified": False,
        "checkpoint_mutated": False,
        "dataset_mutated": False,
    }


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    from ultralytics import YOLO

    yolo_model = YOLO(str(model_path))
    torch_model = yolo_model.model
    torch_model.eval()
    torch_model.to(device)
    return torch_model


def build_weight_delta(preferred_scale: str) -> dict[str, float]:
    ranked = [preferred_scale] + [scale for scale in SCALE_ORDER if scale != preferred_scale]
    values = (1.20, 1.00, 0.95, 0.90)
    return {scale: value for scale, value in zip(ranked, values)}


def initialize_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": "M8_v1c.metric_mining",
        "status": "m8_v1c_metric_mining_dry_run_passed" if args.dry_run else "m8_v1c_metric_mining_completed_no_valid_objects",
        "model_path": str(args.model.resolve()),
        "model_config_path": str(args.model_config.resolve()),
        "data_yaml": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "max_images": args.max_images,
        "device": str(args.device),
        "processed_image_count": 0,
        "processed_object_count": 0,
        "resolved_scale_to_layer": {},
        "warnings": [],
        "size_group_scale_metrics": {},
        "policy_candidate": {},
        "next_allowed_step": "review_m8_v1c_metric_policy_candidate_before_runtime_update",
        "dry_run": bool(args.dry_run),
        "training_executed": False,
        "evaluation_executed": False,
        "dataset_mutated": False,
        "checkpoint_mutated": False,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# M8_v1c XAI Metric Policy Mining",
        "",
        "## Objective",
        "Compute richer per-scale XAI evidence metrics and derive a metric-based policy candidate for size-aware scale weighting.",
        "",
        "## Inputs",
        f"- Model checkpoint: `{summary['model_path']}`",
        f"- Model config: `{summary['model_config_path']}`",
        f"- Dataset YAML: `{summary['data_yaml']}`",
        f"- Split: `{summary['split']}`",
        f"- Resized image size: `{summary['imgsz']}`",
        f"- Dry run: `{summary['dry_run']}`",
        "",
        "## Metrics",
        "- `energy_in_box`: fraction of activation energy that falls inside the GT box",
        "- `background_leakage`: outside-vs-inside activation leakage ratio",
        "- `peak_alignment_score`: how close the hottest point is to the GT box center",
        "- `evidence_score = 0.5 * energy_in_box + 0.3 * peak_alignment_score + 0.2 * (1 - background_leakage)`",
        "",
        "## Processing Summary",
        f"- Processed images: {summary['processed_image_count']}",
        f"- Processed objects: {summary['processed_object_count']}",
        f"- Hook layer mapping: `{json.dumps(summary['resolved_scale_to_layer'], ensure_ascii=True, sort_keys=True)}`",
        "",
        "## Size-group Scale Metrics",
        f"`{json.dumps(summary['size_group_scale_metrics'], ensure_ascii=True, sort_keys=True)}`",
        "",
        "## Policy Candidate",
        f"`{json.dumps(summary['policy_candidate'], ensure_ascii=True, sort_keys=True)}`",
        "",
        "## Caveats",
        "This is still analysis-only and does not prove causal improvement.",
        "No training, validation via Ultralytics APIs, checkpoint mutation, or dataset mutation was performed.",
    ]
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(args: argparse.Namespace) -> None:
    ensure_runtime_dirs()
    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not args.model_config.exists():
        raise FileNotFoundError(f"Model config YAML not found: {args.model_config}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = initialize_summary(args)

    data_config = load_yaml_file(args.data)
    model_config = load_yaml_file(args.model_config)
    class_names = normalize_names(data_config.get("names", {}))
    split_dir = resolve_split_dir(args.data, data_config, args.split)
    image_paths = gather_image_paths(split_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    detect_indices = detect_entry_from_model_yaml(model_config)
    scale_to_layer = map_indices_to_scales(detect_indices)
    summary["resolved_scale_to_layer"] = {scale: int(index) for scale, index in scale_to_layer.items()}

    object_csv_path = args.output_dir / "m8_v1c_xai_metric_object_scores.csv"
    agg_csv_path = args.output_dir / "m8_v1c_scale_evidence_metrics.csv"
    policy_yaml_path = args.output_dir / "m8_v1c_metric_policy_candidate.yaml"
    summary_json_path = args.output_dir / "m8_v1c_metric_summary.json"
    report_path = args.output_dir / "m8_v1c_metric_report.md"
    manifest_path = args.output_dir / "m8_v1c_metric_non_execution_manifest.json"

    if args.dry_run:
        write_csv(object_csv_path, [], CSV_HEADERS)
        write_csv(agg_csv_path, [], [
            "size_group", "scale", "object_count", "mean_energy_in_box", "mean_background_leakage",
            "mean_peak_alignment_score", "mean_evidence_score", "preferred_for_group", "auto_policy_allowed",
        ])
        write_json(summary_json_path, summary)
        report_path.write_text(build_report(summary), encoding="utf-8")
        write_json(manifest_path, build_manifest())
        write_json(policy_yaml_path, {})
        return

    device = torch.device(str(args.device))
    torch_model = load_model(args.model, device)
    collector = ActivationCollector(torch_model, scale_to_layer)
    collector.register()

    object_rows: list[dict[str, Any]] = []
    metric_buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    size_group_scale_object_counts: dict[str, Counter[str]] = defaultdict(Counter)

    try:
        for image_path in image_paths:
            label_path = resolve_label_path(image_path)
            labels = parse_label_file(label_path, class_names, summary["warnings"])
            try:
                image_tensor = load_image_tensor(image_path, args.imgsz, device)
            except Exception as exc:
                summary["warnings"].append(f"Image skipped due to load failure: {image_path} -> {exc}")
                continue

            collector.clear()
            try:
                with torch.no_grad():
                    _ = torch_model(image_tensor)
            except Exception as exc:
                summary["warnings"].append(f"Image skipped due to forward failure: {image_path} -> {exc}")
                continue

            summary["processed_image_count"] += 1
            feature_shapes = {scale: feature_shape_string(collector.activations.get(scale)) for scale in SCALE_ORDER}

            for object_index, label in enumerate(labels):
                area_px = bbox_area_px(label.width, label.height, args.imgsz)
                size_group = size_group_for_area(area_px)
                class_name = class_names.get(label.class_id, f"class_{label.class_id}")
                for scale_name in scale_to_layer:
                    activation = collector.activations.get(scale_name)
                    if activation is None:
                        continue
                    metrics = compute_scale_metrics(activation, label)
                    size_group_scale_object_counts[size_group][scale_name] += 1
                    metric_buckets[f"{size_group}:{scale_name}"]["energy_in_box"].append(metrics["energy_in_box"])
                    metric_buckets[f"{size_group}:{scale_name}"]["background_leakage"].append(metrics["background_leakage"])
                    metric_buckets[f"{size_group}:{scale_name}"]["peak_alignment_score"].append(metrics["peak_alignment_score"])
                    metric_buckets[f"{size_group}:{scale_name}"]["evidence_score"].append(metrics["evidence_score"])

                    object_rows.append(
                        {
                            "image_path": str(image_path.resolve()),
                            "label_path": str(label_path.resolve()),
                            "class_id": label.class_id,
                            "class_name": class_name,
                            "object_index": object_index,
                            "bbox_x_center_norm": label.x_center,
                            "bbox_y_center_norm": label.y_center,
                            "bbox_w_norm": label.width,
                            "bbox_h_norm": label.height,
                            "bbox_area_px": area_px,
                            "object_size_group": size_group,
                            "scale": scale_name,
                            "feature_shape": feature_shapes.get(scale_name, ""),
                            **metrics,
                        }
                    )
                summary["processed_object_count"] += 1
    finally:
        collector.remove()

    agg_rows: list[dict[str, Any]] = []
    policy_candidate: dict[str, Any] = {
        "method_name": "m8_v1c_metric_policy_candidate",
        "policy_source": "xai_metric_mining",
        "split": args.split,
        "size_aware_policy": {},
    }
    size_group_scale_metrics: dict[str, Any] = {}

    for size_group in ("tiny", "small", "medium", "large"):
        per_scale_summary: dict[str, Any] = {}
        evidence_ranking: list[tuple[str, float]] = []
        for scale_name in scale_to_layer:
            bucket = metric_buckets.get(f"{size_group}:{scale_name}", {})
            object_count = size_group_scale_object_counts[size_group].get(scale_name, 0)
            if object_count <= 0:
                continue
            mean_energy = mean(bucket["energy_in_box"])
            mean_leakage = mean(bucket["background_leakage"])
            mean_peak = mean(bucket["peak_alignment_score"])
            mean_evidence = mean(bucket["evidence_score"])
            per_scale_summary[scale_name] = {
                "object_count": object_count,
                "mean_energy_in_box": mean_energy,
                "mean_background_leakage": mean_leakage,
                "mean_peak_alignment_score": mean_peak,
                "mean_evidence_score": mean_evidence,
            }
            evidence_ranking.append((scale_name, mean_evidence))

        size_group_scale_metrics[size_group] = per_scale_summary
        if not evidence_ranking:
            policy_candidate["size_aware_policy"][size_group] = {
                "mode": "identity",
                "reason": "no_metric_evidence_available",
            }
            continue

        evidence_ranking.sort(key=lambda item: (-item[1], item[0]))
        best_scale, best_score = evidence_ranking[0]
        second_score = evidence_ranking[1][1] if len(evidence_ranking) > 1 else 0.0
        margin = best_score - second_score
        total_objects = max(size_group_scale_object_counts[size_group].values(), default=0)
        auto_policy_allowed = (
            size_group in {"small", "medium", "large"}
            and total_objects >= args.min_objects_for_policy
            and margin >= args.min_best_minus_second_margin
        )
        if auto_policy_allowed:
            policy_candidate["size_aware_policy"][size_group] = {
                "mode": "preferred_scale",
                "preferred_scale": best_scale,
                "weight_delta": build_weight_delta(best_scale),
                "best_minus_second_evidence_margin": margin,
                "object_count": total_objects,
            }
        else:
            policy_candidate["size_aware_policy"][size_group] = {
                "mode": "identity",
                "best_candidate_scale": best_scale,
                "best_minus_second_evidence_margin": margin,
                "object_count": total_objects,
                "reason": "insufficient_metric_margin_or_sample_count",
            }

        for scale_name, scale_summary in per_scale_summary.items():
            agg_rows.append(
                {
                    "size_group": size_group,
                    "scale": scale_name,
                    "object_count": scale_summary["object_count"],
                    "mean_energy_in_box": scale_summary["mean_energy_in_box"],
                    "mean_background_leakage": scale_summary["mean_background_leakage"],
                    "mean_peak_alignment_score": scale_summary["mean_peak_alignment_score"],
                    "mean_evidence_score": scale_summary["mean_evidence_score"],
                    "preferred_for_group": best_scale,
                    "auto_policy_allowed": auto_policy_allowed,
                }
            )

    policy_candidate["size_aware_policy"]["unknown"] = {
        "mode": "identity",
        "reason": "fail_closed_for_unknown_or_ambiguous_size_group",
    }
    summary["size_group_scale_metrics"] = size_group_scale_metrics
    summary["policy_candidate"] = policy_candidate
    summary["status"] = (
        "m8_v1c_metric_mining_completed" if summary["processed_object_count"] > 0 else "m8_v1c_metric_mining_completed_no_valid_objects"
    )

    write_csv(object_csv_path, object_rows, CSV_HEADERS)
    write_csv(
        agg_csv_path,
        agg_rows,
        [
            "size_group",
            "scale",
            "object_count",
            "mean_energy_in_box",
            "mean_background_leakage",
            "mean_peak_alignment_score",
            "mean_evidence_score",
            "preferred_for_group",
            "auto_policy_allowed",
        ],
    )
    write_json(summary_json_path, summary)
    report_path.write_text(build_report(summary), encoding="utf-8")
    write_json(manifest_path, build_manifest())
    with policy_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(policy_candidate, handle, sort_keys=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
