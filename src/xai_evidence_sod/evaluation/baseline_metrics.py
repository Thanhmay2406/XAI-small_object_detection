"""Baseline evaluation artifact builders."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class DetectionSample:
    """One predicted detection sample for analysis."""

    image_path: str
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def summarize_validation_results(results: Any, class_names: list[str]) -> dict[str, Any]:
    """Convert Ultralytics validation output into a serializable summary."""

    box = results.box
    overall = {
        "map": _safe_float(box.map),
        "map50": _safe_float(box.map50),
        "map75": _safe_float(box.map75),
        "maps": [_safe_float(value) for value in box.maps.tolist()],
        "fitness": _safe_float(results.fitness),
    }
    per_class = []
    for class_id, class_name in enumerate(class_names):
        class_map = overall["maps"][class_id] if class_id < len(overall["maps"]) else None
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "map": class_map,
            }
        )
    return {
        "overall": overall,
        "per_class": per_class,
        "save_dir": str(results.save_dir),
        "speed": dict(results.speed),
    }


def collect_prediction_samples(prediction_results: list[Any]) -> list[DetectionSample]:
    """Flatten prediction outputs into serializable rows."""

    rows: list[DetectionSample] = []
    for result in prediction_results:
        names = result.names
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.tolist()
        confs = result.boxes.conf.tolist()
        classes = result.boxes.cls.tolist()
        for coords, conf, class_id in zip(xyxy, confs, classes):
            rows.append(
                DetectionSample(
                    image_path=str(result.path),
                    class_id=int(class_id),
                    class_name=names[int(class_id)],
                    confidence=float(conf),
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                )
            )
    return rows


def write_prediction_csv(rows: list[DetectionSample], output_path: str | Path) -> Path:
    """Write prediction rows to CSV."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return output_path


def analyze_class_error_cases(
    dataset_yaml_path: str | Path,
    prediction_results: list[Any],
    target_class_name: str,
    output_csv_path: str | Path,
    iou_threshold: float = 0.5,
) -> Path:
    """Analyze false negatives for a target class using YOLO-format labels."""

    dataset_config = _load_yaml(dataset_yaml_path)
    class_lookup = _normalize_names(dataset_config["names"])
    if target_class_name not in class_lookup.values():
        raise ValueError(f"Target class {target_class_name!r} not found in dataset names")
    target_class_id = next(class_id for class_id, name in class_lookup.items() if name == target_class_name)

    rows: list[dict[str, Any]] = []
    for result in prediction_results:
        image_path = Path(result.path).resolve()
        width, height = Image.open(image_path).size
        gt_rows = _load_label_rows(image_path, dataset_config, target_class_id, width, height)
        predictions = _extract_class_predictions(result, target_class_id)
        for gt_index, gt_box in enumerate(gt_rows):
            best_iou = 0.0
            best_conf = 0.0
            matched = False
            for pred_box in predictions:
                iou = _xyxy_iou(gt_box, pred_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_conf = pred_box[4]
                if iou >= iou_threshold:
                    matched = True
            if not matched:
                rows.append(
                    {
                        "image_path": str(image_path),
                        "target_class": target_class_name,
                        "gt_index": gt_index,
                        "best_iou": round(best_iou, 6),
                        "best_confidence": round(best_conf, 6),
                        "error_type": "false_negative",
                    }
                )

    output_path = Path(output_csv_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "target_class", "gt_index", "best_iou", "best_confidence", "error_type"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_markdown_report(
    metrics_summary: dict[str, Any],
    target_class_name: str,
    error_case_csv: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a concise markdown baseline evaluation report."""

    overall = metrics_summary["overall"]
    per_class = metrics_summary["per_class"]
    error_count = _count_csv_rows(error_case_csv)
    lines = [
        "# Baseline Evaluation Report",
        "",
        "## Overall Metrics",
        "",
        f"- mAP50-95: {overall['map']:.4f}" if overall["map"] is not None else "- mAP50-95: unavailable",
        f"- mAP50: {overall['map50']:.4f}" if overall["map50"] is not None else "- mAP50: unavailable",
        f"- mAP75: {overall['map75']:.4f}" if overall["map75"] is not None else "- mAP75: unavailable",
        "",
        "## Per-Class Metrics",
        "",
        "| class_id | class_name | map |",
        "| --- | --- | --- |",
    ]
    for row in per_class:
        map_value = "n/a" if row["map"] is None else f"{row['map']:.4f}"
        lines.append(f"| {row['class_id']} | {row['class_name']} | {map_value} |")
    lines.extend(
        [
            "",
            f"## {target_class_name} Error Cases",
            "",
            f"- False-negative rows saved: {error_count}",
            f"- Error case CSV: `{Path(error_case_csv).resolve()}`",
            "",
        ]
    )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def dump_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write JSON to disk."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _safe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise ValueError("Unsupported names structure in dataset yaml")


def _load_label_rows(image_path: Path, dataset_config: dict[str, Any], target_class_id: int, width: int, height: int) -> list[tuple[float, float, float, float]]:
    dataset_root = image_path.parents[2]
    split_name = image_path.parent.name
    label_path = dataset_root / "labels" / split_name / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes = []
    for row in text.splitlines():
        parts = row.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        if class_id != target_class_id:
            continue
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        x1 = (x_center - box_width / 2.0) * width
        y1 = (y_center - box_height / 2.0) * height
        x2 = (x_center + box_width / 2.0) * width
        y2 = (y_center + box_height / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def _extract_class_predictions(result: Any, target_class_id: int) -> list[tuple[float, float, float, float, float]]:
    predictions: list[tuple[float, float, float, float, float]] = []
    if result.boxes is None:
        return predictions
    for coords, conf, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        if int(class_id) != target_class_id:
            continue
        predictions.append((float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), float(conf)))
    return predictions


def _xyxy_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def _count_csv_rows(path: str | Path) -> int:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)
