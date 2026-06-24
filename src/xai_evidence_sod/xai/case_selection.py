"""Curated XAI case selection for post-hoc baseline evidence analysis."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from xai_evidence_sod.data import load_yolo_dataset_config


@dataclass(frozen=True)
class CuratedCase:
    """One curated image-level case for XAI evidence extraction."""

    case_id: str
    image_path: str
    split: str
    focus_class: str
    case_group: str
    source_role: str
    gt_class_name: str
    pred_class_name: str
    gt_index: str
    pred_index: str
    best_iou: float
    confidence: float | None
    size_group: str
    pixel_size_group: str
    pixel_area: float | None
    gt_x1: float | None
    gt_y1: float | None
    gt_x2: float | None
    gt_y2: float | None
    pred_x1: float | None
    pred_y1: float | None
    pred_x2: float | None
    pred_y2: float | None
    is_near_threshold: bool
    notes: str


def build_curated_cases(
    cases_csv: str | Path,
    data_path: str | Path,
    prediction_rows_csv: str | Path,
    focus_class: str,
    max_cases: int | None,
    seed: int,
    imgsz: int = 640,
    iou_threshold: float = 0.5,
    near_threshold_iou: float = 0.4,
) -> list[CuratedCase]:
    """Build a balanced curated case list, including TP proxies when possible."""

    error_cases = _load_error_cases(cases_csv, focus_class, near_threshold_iou)
    tp_cases = _build_true_positive_cases(
        data_path=data_path,
        prediction_rows_csv=prediction_rows_csv,
        focus_class=focus_class,
        imgsz=imgsz,
        iou_threshold=iou_threshold,
        near_threshold_iou=near_threshold_iou,
    )
    all_cases = tp_cases + error_cases
    if max_cases is None or max_cases >= len(all_cases):
        return all_cases
    return _balanced_sample(all_cases, max_cases, seed)


def summarize_case_groups(cases: list[CuratedCase]) -> dict[str, int]:
    """Count curated cases by group."""

    counter = Counter(case.case_group for case in cases)
    if any(case.is_near_threshold for case in cases):
        counter["near_threshold_overlap"] = sum(case.is_near_threshold for case in cases)
    return dict(counter)


def _load_error_cases(cases_csv: str | Path, focus_class: str, near_threshold_iou: float) -> list[CuratedCase]:
    rows: list[CuratedCase] = []
    with Path(cases_csv).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            case_group = row["primary_error_type"]
            rows.append(
                CuratedCase(
                    case_id=f"error_{index:04d}",
                    image_path=row["image_path"],
                    split=row.get("split", ""),
                    focus_class=focus_class,
                    case_group=case_group,
                    source_role=row.get("source_role", ""),
                    gt_class_name=row.get("gt_class_name", ""),
                    pred_class_name=row.get("pred_class_name", ""),
                    gt_index=row.get("gt_index", ""),
                    pred_index=row.get("pred_index", ""),
                    best_iou=_safe_float(row.get("best_iou")) or 0.0,
                    confidence=_safe_float(row.get("confidence")),
                    size_group=row.get("size_group", ""),
                    pixel_size_group=row.get("pixel_size_group", ""),
                    pixel_area=_safe_float(row.get("pixel_area")),
                    gt_x1=_safe_float(row.get("gt_x1")),
                    gt_y1=_safe_float(row.get("gt_y1")),
                    gt_x2=_safe_float(row.get("gt_x2")),
                    gt_y2=_safe_float(row.get("gt_y2")),
                    pred_x1=_safe_float(row.get("pred_x1")),
                    pred_y1=_safe_float(row.get("pred_y1")),
                    pred_x2=_safe_float(row.get("pred_x2")),
                    pred_y2=_safe_float(row.get("pred_y2")),
                    is_near_threshold=(_safe_float(row.get("best_iou")) or 0.0) >= near_threshold_iou,
                    notes=row.get("notes", ""),
                )
            )
    return rows


def _build_true_positive_cases(
    data_path: str | Path,
    prediction_rows_csv: str | Path,
    focus_class: str,
    imgsz: int,
    iou_threshold: float,
    near_threshold_iou: float,
) -> list[CuratedCase]:
    config = load_yolo_dataset_config(data_path)
    image_index = {image_path.name: image_path for split in config.splits for image_path in split.image_files}
    pred_by_image = _load_focus_predictions(prediction_rows_csv, image_index, focus_class)

    rows: list[CuratedCase] = []
    case_index = 0
    for split in config.splits:
        if split.name != "test":
            continue
        image_map = {path.stem: path for path in split.image_files}
        for label_path in split.label_files:
            image_path = image_map.get(label_path.stem)
            if image_path is None:
                continue
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            width_px, height_px = Image.open(image_path).size
            gt_boxes = _load_focus_gt_boxes(text, width_px, height_px, focus_class, config.names, imgsz)
            if not gt_boxes:
                continue
            pred_boxes = pred_by_image.get(str(image_path.resolve()), [])
            matches = _match_focus_boxes(gt_boxes, pred_boxes, iou_threshold)
            for gt_idx, pred_idx, best_iou in matches:
                gt_box = gt_boxes[gt_idx]
                pred_box = pred_boxes[pred_idx]
                rows.append(
                    CuratedCase(
                        case_id=f"tp_{case_index:04d}",
                        image_path=str(image_path.resolve()),
                        split=split.name,
                        focus_class=focus_class,
                        case_group="true_positive_proxy",
                        source_role="matched_pair",
                        gt_class_name=focus_class,
                        pred_class_name=focus_class,
                        gt_index=str(gt_idx),
                        pred_index=str(pred_idx),
                        best_iou=best_iou,
                        confidence=pred_box["confidence"],
                        size_group=gt_box["size_group"],
                        pixel_size_group=gt_box["pixel_size_group"],
                        pixel_area=gt_box["pixel_area"],
                        gt_x1=gt_box["x1"],
                        gt_y1=gt_box["y1"],
                        gt_x2=gt_box["x2"],
                        gt_y2=gt_box["y2"],
                        pred_x1=pred_box["x1"],
                        pred_y1=pred_box["y1"],
                        pred_x2=pred_box["x2"],
                        pred_y2=pred_box["y2"],
                        is_near_threshold=best_iou >= near_threshold_iou,
                        notes="Matched focus-class GT/pred pair used as a true-positive proxy for evidence comparison.",
                    )
                )
                case_index += 1
    return rows


def _load_focus_predictions(prediction_rows_csv: str | Path, image_index: dict[str, Path], focus_class: str) -> dict[str, list[dict[str, float | str]]]:
    pred_by_image: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    with Path(prediction_rows_csv).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["class_name"] != focus_class:
                continue
            image_name = Path(row["image_path"]).name
            image_path = image_index.get(image_name)
            if image_path is None:
                continue
            pred_by_image[str(image_path.resolve())].append(
                {
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "confidence": float(row["confidence"]),
                }
            )
    return pred_by_image


def _load_focus_gt_boxes(
    text: str,
    width_px: int,
    height_px: int,
    focus_class: str,
    class_names: tuple[str, ...],
    imgsz: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for raw_line in text.splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        class_name = class_names[class_id]
        if class_name != focus_class:
            continue
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        x1 = (x_center - box_width / 2.0) * width_px
        y1 = (y_center - box_height / 2.0) * height_px
        x2 = (x_center + box_width / 2.0) * width_px
        y2 = (y_center + box_height / 2.0) * height_px
        pixel_width = box_width * float(imgsz)
        pixel_height = box_height * float(imgsz)
        pixel_area = pixel_width * pixel_height
        rows.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "size_group": _assign_size_group(box_width * box_height),
                "pixel_size_group": _assign_pixel_size_group(pixel_area),
                "pixel_area": pixel_area,
            }
        )
    return rows


def _match_focus_boxes(
    gt_boxes: list[dict[str, float | str]],
    pred_boxes: list[dict[str, float | str]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            iou = _xyxy_iou(
                (float(gt_box["x1"]), float(gt_box["y1"]), float(gt_box["x2"]), float(gt_box["y2"])),
                (float(pred_box["x1"]), float(pred_box["y1"]), float(pred_box["x2"]), float(pred_box["y2"])),
            )
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))
    return matches


def _balanced_sample(cases: list[CuratedCase], max_cases: int, seed: int) -> list[CuratedCase]:
    rng = random.Random(seed)
    by_group: dict[str, list[CuratedCase]] = defaultdict(list)
    for case in cases:
        by_group[case.case_group].append(case)
    for group_cases in by_group.values():
        rng.shuffle(group_cases)

    ordered_groups = ["true_positive_proxy", "localization_error", "false_negative", "false_positive"]
    sampled: list[CuratedCase] = []
    while len(sampled) < max_cases:
        progressed = False
        for group in ordered_groups:
            group_cases = by_group.get(group, [])
            if not group_cases:
                continue
            sampled.append(group_cases.pop())
            progressed = True
            if len(sampled) >= max_cases:
                break
        if not progressed:
            break

    if len(sampled) < max_cases:
        leftovers = [case for group_cases in by_group.values() for case in group_cases]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: max_cases - len(sampled)])

    sampled.sort(key=lambda case: (case.case_group, case.image_path, case.case_id))
    return sampled[:max_cases]


def _assign_size_group(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def _assign_pixel_size_group(pixel_area: float) -> str:
    if pixel_area < 16 * 16:
        return "tiny_px"
    if pixel_area < 32 * 32:
        return "small_px"
    if pixel_area < 96 * 96:
        return "medium_px"
    return "large_px"


def _xyxy_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _safe_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)
