"""M8.0 diagnostic-only script for activation-guided FPN scale diagnosis.

This script analyzes which YOLOv8 or YOLOv8-P2 FPN scale concentrates the most
activation energy inside ground-truth defect boxes. It does not train, run
official validation, export predictions, mutate checkpoints, or modify dataset
artifacts. It only performs direct forward passes on the loaded torch model to
capture intermediate feature maps for evidence gathering.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml
from PIL import Image
import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_HEADERS = [
    "image_path",
    "label_path",
    "class_id",
    "class_name",
    "bbox_x_center_norm",
    "bbox_y_center_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "bbox_area_px",
    "object_size_group",
    "dominant_scale",
    "P2_score",
    "P3_score",
    "P4_score",
    "P5_score",
    "P2_feature_shape",
    "P3_feature_shape",
    "P4_feature_shape",
    "P5_feature_shape",
]
SCALE_ORDER = ["P2", "P3", "P4", "P5"]


@dataclass
class LabelRecord:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M8.0 diagnostic-only XAI-guided FPN scale selection analysis."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to trained YOLO checkpoint .pt.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional YOLO model YAML used to resolve Detect input layers.",
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset YAML.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to analyze.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square resize size used for the diagnostic.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=64,
        help="Maximum number of images to analyze. Use 0 or negative for the full split.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu, cuda:0, or 0 depending on the local setup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/m8_xai_fpn_scale_selection"),
        help="Directory for CSV, summary, report, and manifest outputs.",
    )
    parser.add_argument(
        "--candidate-layer-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit layer indices to hook in order from highest resolution to lowest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, load model structure, resolve hook layers, and write non-execution artifacts only.",
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
    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


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
    if cv2 is not None:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image with cv2: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    else:
        if Image is None:
            raise RuntimeError("Neither cv2 nor PIL is available for image loading.")
        with Image.open(image_path) as image:
            resized = image.convert("RGB").resize((imgsz, imgsz))
            resized = torch.ByteTensor(torch.ByteStorage.from_buffer(resized.tobytes())).view(imgsz, imgsz, 3).numpy()

    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor.unsqueeze(0).to(device)


def parse_label_file(
    label_path: Path,
    class_names: dict[int, str],
    warnings: list[str],
) -> list[LabelRecord]:
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
        records.append(
            LabelRecord(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return records


def detect_entry_from_model_yaml(model_config: dict[str, Any]) -> tuple[list[int], bool]:
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
        raise ValueError("Detect layer 'from' field must be a list of layer indices.")

    if len(layer_from) not in {3, 4}:
        raise ValueError("Detect layer 'from' field must contain either 3 or 4 indices.")

    try:
        return [int(index) for index in layer_from], True
    except (TypeError, ValueError) as exc:
        raise ValueError("Detect layer 'from' field contains non-integer values.") from exc


def map_indices_to_scales(indices: list[int]) -> dict[str, int]:
    if len(indices) == 4:
        scales = ["P2", "P3", "P4", "P5"]
    elif len(indices) == 3:
        scales = ["P3", "P4", "P5"]
    else:
        raise ValueError("Expected either 3 or 4 layer indices.")
    return dict(zip(scales, indices, strict=True))


def normalize_candidate_layers(candidate_layer_indices: list[int]) -> dict[str, int]:
    if len(candidate_layer_indices) == 4:
        scales = ["P2", "P3", "P4", "P5"]
    elif len(candidate_layer_indices) == 3:
        scales = ["P3", "P4", "P5"]
    else:
        raise ValueError("--candidate-layer-indices must contain either 3 or 4 integers.")
    return dict(zip(scales, candidate_layer_indices, strict=True))


class ActivationCollector:
    def __init__(self, torch_model: torch.nn.Module, scale_to_layer: dict[str, int]) -> None:
        self.torch_model = torch_model
        self.scale_to_layer = scale_to_layer
        self.activations: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.layer_count = len(getattr(torch_model, "model", []))

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
                raise ValueError(
                    f"Invalid layer index {layer_index} for {scale_name}. Model has {len(module_list)} layers."
                )

            def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any, scale: str = scale_name) -> None:
                tensor = self._extract_4d_tensor(output)
                if tensor is not None:
                    self.activations[scale] = tensor

            self.handles.append(module_list[layer_index].register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def auto_detect_scale_mapping(
    torch_model: torch.nn.Module,
    image_tensor: torch.Tensor,
    warnings: list[str],
) -> dict[str, int]:
    module_list = getattr(torch_model, "model", None)
    if module_list is None:
        raise ValueError("Loaded torch model does not expose a .model module list.")

    tail_indices = list(range(max(0, len(module_list) - 16), len(module_list)))
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer_index: int):
        def hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
            tensor: torch.Tensor | None = None
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                tensor = output.detach().cpu()
            elif isinstance(output, (list, tuple)):
                for item in output:
                    if isinstance(item, torch.Tensor) and item.ndim == 4:
                        tensor = item.detach().cpu()
                        break
            if tensor is not None:
                captured[layer_index] = tensor

        return hook

    for layer_index in tail_indices:
        handles.append(module_list[layer_index].register_forward_hook(make_hook(layer_index)))

    try:
        with torch.no_grad():
            _ = torch_model(image_tensor)
    finally:
        for handle in handles:
            handle.remove()

    if not captured:
        raise RuntimeError("No 4D feature maps were captured. Try --candidate-layer-indices or --model-config.")

    resolution_to_layer: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
    for layer_index, tensor in captured.items():
        _, _, height, width = tensor.shape
        resolution = (height, width)
        existing = resolution_to_layer.get(resolution)
        if existing is None or layer_index > existing[0]:
            resolution_to_layer[resolution] = (layer_index, tensor)

    sorted_candidates = sorted(
        (
            {
                "layer_index": layer_index,
                "shape": tuple(tensor.shape),
                "resolution": resolution,
                "area": resolution[0] * resolution[1],
            }
            for resolution, (layer_index, tensor) in resolution_to_layer.items()
        ),
        key=lambda item: (item["area"], item["layer_index"]),
        reverse=True,
    )

    if len(sorted_candidates) > 4:
        warnings.append(
            "Auto-detect captured more than 4 unique 4D feature resolutions; selecting the 4 largest resolutions."
        )
        sorted_candidates = sorted_candidates[:4]

    if len(sorted_candidates) < 3:
        raise RuntimeError("No 4D feature maps were captured. Try --candidate-layer-indices or --model-config.")

    chosen_indices = [int(item["layer_index"]) for item in sorted_candidates]
    return normalize_candidate_layers(chosen_indices)


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


def score_scale_activation(
    activation: torch.Tensor,
    label: LabelRecord,
) -> float:
    if activation.ndim != 4:
        return 0.0

    feature = activation[0]
    energy = feature.abs().mean(dim=0)
    height, width = int(energy.shape[0]), int(energy.shape[1])

    x1 = int((label.x_center - (label.width / 2.0)) * width)
    y1 = int((label.y_center - (label.height / 2.0)) * height)
    x2 = int((label.x_center + (label.width / 2.0)) * width)
    y2 = int((label.y_center + (label.height / 2.0)) * height)

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    mask = torch.zeros((height, width), dtype=energy.dtype)
    mask[y1:y2, x1:x2] = 1.0
    denominator = float(energy.sum().item()) + 1e-8
    numerator = float((energy * mask).sum().item())
    return numerator / denominator


def feature_shape_string(activation: torch.Tensor | None) -> str:
    if activation is None:
        return ""
    return "x".join(str(dim) for dim in activation.shape)


def dominant_scale_from_scores(scale_scores: dict[str, float]) -> str:
    if not scale_scores:
        return ""
    return max(scale_scores.items(), key=lambda item: item[1])[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest() -> dict[str, Any]:
    return {
        "phase": "M8.0",
        "script_purpose": "diagnostic_only",
        "training_executed": False,
        "evaluation_executed": False,
        "ultralytics_train_api_called": False,
        "ultralytics_val_api_called": False,
        "ultralytics_predict_api_called": False,
        "model_weights_modified": False,
        "dataset_files_modified": False,
        "labels_modified": False,
        "predictions_exported_for_training": False,
        "checkpoint_mutated": False,
        "dataset_mutated": False,
    }


def automatic_conclusion(scale_counts: Counter[str]) -> str:
    high_res = scale_counts.get("P2", 0) + scale_counts.get("P3", 0)
    low_res = scale_counts.get("P4", 0) + scale_counts.get("P5", 0)
    if high_res > low_res:
        return "High-resolution FPN levels appear important for this split."
    if low_res > high_res:
        return "Larger/contextual feature levels dominate this diagnostic run."
    return "No clear scale dominance was observed."


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# M8.0 XAI-guided FPN Scale Selection Diagnosis",
        "",
        "## Objective",
        "Assess which FPN scale concentrates activation most strongly inside ground-truth defect boxes under a resize-based diagnostic forward pass.",
        "",
        "## Inputs",
        f"- Model checkpoint: `{summary['model_path']}`",
        f"- Model config: `{summary['model_config_path']}`" if summary["model_config_path"] else "- Model config: not provided",
        f"- Dataset YAML: `{summary['data_yaml']}`",
        f"- Split: `{summary['split']}`",
        f"- Resized image size: `{summary['imgsz']}`",
        f"- Dry run: `{summary['dry_run']}`",
        "",
        "## Processing Summary",
        f"- Processed images: {summary['processed_image_count']}",
        f"- Processed objects: {summary['processed_object_count']}",
        f"- Hook layer mapping: `{json.dumps(summary['resolved_scale_to_layer'], ensure_ascii=True, sort_keys=True)}`",
        "",
        "## Score Formula",
        "For each ground-truth object and each scale s:",
        "",
        "```text",
        "score_s = sum(abs(feature_s) * bbox_mask_s) / (sum(abs(feature_s)) + eps)",
        "```",
        "",
        "## Dominant Scale Counts",
        f"`{json.dumps(summary['scale_counts'], ensure_ascii=True, sort_keys=True)}`",
        "",
        "## Mean Score Per Scale",
        f"`{json.dumps(summary['scale_mean_scores'], ensure_ascii=True, sort_keys=True)}`",
        "",
        "## Object Size Group Summary",
        f"`{json.dumps(summary['object_size_group_to_dominant_scale'], ensure_ascii=True, sort_keys=True)}`",
        "",
        "## Automatic Conclusion",
        automatic_conclusion(Counter(summary["scale_counts"])),
        "",
        "## Caveats",
        "This diagnostic uses direct square resize to the requested image size and is activation-guided only.",
        "This is activation-guided diagnosis only. It is not training-time reweighting and does not prove causal improvement.",
        "No training, validation, prediction export, checkpoint mutation, dataset mutation, or label mutation was performed.",
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


def load_model(model_path: Path, device: torch.device) -> tuple[Any, torch.nn.Module]:
    from ultralytics import YOLO

    yolo_model = YOLO(str(model_path))
    torch_model = yolo_model.model
    torch_model.eval()
    torch_model.to(device)
    return yolo_model, torch_model


def initialize_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": "M8.0",
        "status": "m8_dry_run_passed" if args.dry_run else "m8_scale_diagnosis_completed_no_valid_objects",
        "model_path": str(args.model.resolve()),
        "model_config_path": str(args.model_config.resolve()) if args.model_config else None,
        "data_yaml": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "max_images": args.max_images,
        "device": str(args.device),
        "processed_image_count": 0,
        "processed_object_count": 0,
        "scale_counts": {},
        "scale_mean_scores": {},
        "object_size_group_counts": {},
        "object_size_group_to_dominant_scale": {},
        "candidate_layer_indices": [],
        "resolved_from_model_config": False,
        "resolved_scale_to_layer": {},
        "resolved_scale_names": [],
        "warnings": [],
        "training_executed": False,
        "evaluation_executed": False,
        "dataset_mutated": False,
        "checkpoint_mutated": False,
        "next_allowed_step": "review_m8_scale_diagnosis_before_training_time_reweighting",
        "dry_run": bool(args.dry_run),
    }


def resolve_scale_layers(
    args: argparse.Namespace,
    torch_model: torch.nn.Module,
    summary: dict[str, Any],
    sample_image_tensor: torch.Tensor | None,
) -> dict[str, int]:
    if args.candidate_layer_indices:
        scale_to_layer = normalize_candidate_layers(args.candidate_layer_indices)
    elif args.model_config:
        model_config = load_yaml_file(args.model_config)
        indices, resolved = detect_entry_from_model_yaml(model_config)
        scale_to_layer = map_indices_to_scales(indices)
        summary["resolved_from_model_config"] = resolved
    else:
        if sample_image_tensor is None:
            raise RuntimeError("Auto-detect fallback requires at least one sample image.")
        scale_to_layer = auto_detect_scale_mapping(torch_model, sample_image_tensor, summary["warnings"])

    module_list = getattr(torch_model, "model", None)
    if module_list is None:
        raise ValueError("Loaded torch model does not expose a .model module list.")

    for scale_name, layer_index in scale_to_layer.items():
        if layer_index < 0 or layer_index >= len(module_list):
            raise ValueError(f"Invalid layer index {layer_index} for {scale_name}.")

    summary["candidate_layer_indices"] = [int(index) for index in scale_to_layer.values()]
    summary["resolved_scale_to_layer"] = {key: int(value) for key, value in scale_to_layer.items()}
    summary["resolved_scale_names"] = list(scale_to_layer.keys())
    return scale_to_layer


def run_analysis(args: argparse.Namespace) -> None:
    ensure_runtime_dirs()

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.model_config and not args.model_config.exists():
        raise FileNotFoundError(f"Model config YAML not found: {args.model_config}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = initialize_summary(args)
    data_config = load_yaml_file(args.data)
    class_names = normalize_names(data_config.get("names", {}))
    split_dir = resolve_split_dir(args.data, data_config, args.split)
    image_paths = gather_image_paths(split_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    device = torch.device(str(args.device))
    _, torch_model = load_model(args.model, device)

    sample_image_tensor: torch.Tensor | None = None
    if image_paths and not args.dry_run:
        sample_image_tensor = load_image_tensor(image_paths[0], args.imgsz, device)
    elif image_paths and args.model_config is None and not args.candidate_layer_indices:
        sample_image_tensor = load_image_tensor(image_paths[0], args.imgsz, device)

    scale_to_layer = resolve_scale_layers(args, torch_model, summary, sample_image_tensor)

    csv_path = args.output_dir / "m8_scale_object_scores.csv"
    summary_path = args.output_dir / "m8_scale_summary.json"
    report_path = args.output_dir / "m8_scale_selection_report.md"
    manifest_path = args.output_dir / "m8_non_execution_manifest.json"

    if args.dry_run:
        write_csv(csv_path, [])
        write_json(summary_path, summary)
        report_path.write_text(build_report(summary), encoding="utf-8")
        write_json(manifest_path, build_manifest())
        return

    collector = ActivationCollector(torch_model, scale_to_layer)
    collector.register()

    rows: list[dict[str, Any]] = []
    scale_counts: Counter[str] = Counter()
    object_size_group_counts: Counter[str] = Counter()
    size_group_scale_counts: dict[str, Counter[str]] = defaultdict(Counter)
    score_buckets: dict[str, list[float]] = defaultdict(list)

    try:
        for image_path in image_paths:
            label_path = resolve_label_path(image_path)
            labels = parse_label_file(label_path, class_names, summary["warnings"])
            try:
                image_tensor = load_image_tensor(image_path, args.imgsz, device)
            except Exception as exc:  # pragma: no cover - runtime robustness branch
                summary["warnings"].append(f"Image skipped due to load failure: {image_path} -> {exc}")
                continue

            collector.clear()
            try:
                with torch.no_grad():
                    _ = torch_model(image_tensor)
            except Exception as exc:  # pragma: no cover - runtime robustness branch
                summary["warnings"].append(f"Image skipped due to forward failure: {image_path} -> {exc}")
                continue

            summary["processed_image_count"] += 1
            if not collector.activations:
                collector.remove()
                raise RuntimeError(
                    "No 4D feature maps were captured. Try --candidate-layer-indices or --model-config."
                )

            feature_shapes = {
                scale_name: feature_shape_string(collector.activations.get(scale_name))
                for scale_name in SCALE_ORDER
            }

            for label in labels:
                per_scale_scores: dict[str, float] = {}
                for scale_name in scale_to_layer:
                    activation = collector.activations.get(scale_name)
                    if activation is None:
                        continue
                    per_scale_scores[scale_name] = score_scale_activation(activation, label)
                    score_buckets[scale_name].append(per_scale_scores[scale_name])

                dominant_scale = dominant_scale_from_scores(per_scale_scores)
                area_px = bbox_area_px(label.width, label.height, args.imgsz)
                size_group = size_group_for_area(area_px)

                if dominant_scale:
                    scale_counts[dominant_scale] += 1
                    size_group_scale_counts[size_group][dominant_scale] += 1
                object_size_group_counts[size_group] += 1
                summary["processed_object_count"] += 1

                class_name = class_names.get(label.class_id, f"class_{label.class_id}")
                row: dict[str, Any] = {
                    "image_path": str(image_path.resolve()),
                    "label_path": str(label_path.resolve()),
                    "class_id": label.class_id,
                    "class_name": class_name,
                    "bbox_x_center_norm": label.x_center,
                    "bbox_y_center_norm": label.y_center,
                    "bbox_w_norm": label.width,
                    "bbox_h_norm": label.height,
                    "bbox_area_px": area_px,
                    "object_size_group": size_group,
                    "dominant_scale": dominant_scale,
                }
                for scale_name in SCALE_ORDER:
                    row[f"{scale_name}_score"] = per_scale_scores.get(scale_name, "")
                    row[f"{scale_name}_feature_shape"] = feature_shapes.get(scale_name, "")
                rows.append(row)
    finally:
        collector.remove()

    summary["scale_counts"] = dict(scale_counts)
    summary["scale_mean_scores"] = {
        scale_name: mean(values) for scale_name, values in sorted(score_buckets.items()) if values
    }
    summary["object_size_group_counts"] = dict(object_size_group_counts)
    summary["object_size_group_to_dominant_scale"] = {
        size_group: {
            "dominant_scale": dominant_scale_from_scores(
                {scale_name: float(count) for scale_name, count in counts.items()}
            ),
            "counts": dict(counts),
            "total_objects": int(sum(counts.values())),
        }
        for size_group, counts in sorted(size_group_scale_counts.items())
    }

    if summary["processed_object_count"] > 0:
        summary["status"] = "m8_scale_diagnosis_completed"
    else:
        summary["status"] = "m8_scale_diagnosis_completed_no_valid_objects"

    write_csv(csv_path, rows)
    write_json(summary_path, summary)
    report_path.write_text(build_report(summary), encoding="utf-8")
    write_json(manifest_path, build_manifest())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
