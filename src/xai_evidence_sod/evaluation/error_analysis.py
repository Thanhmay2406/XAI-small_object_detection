"""Baseline error analysis helpers for post-evaluation research review."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from xai_evidence_sod.data import load_yolo_dataset_config
from xai_evidence_sod.utils.config import ensure_dir, ensure_file


@dataclass(frozen=True)
class GroundTruthBox:
    """One ground-truth bounding box resolved into pixel space."""

    image_path: str
    split: str
    class_id: int
    class_name: str
    gt_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    area_ratio: float
    aspect_ratio: float
    size_group: str
    pixel_width: float
    pixel_height: float
    pixel_area: float
    pixel_size_group: str


@dataclass(frozen=True)
class PredictionBox:
    """One predicted bounding box read from exported evaluation rows."""

    image_path: str
    class_id: int
    class_name: str
    pred_index: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    area: float


def run_baseline_error_analysis(
    eval_dir: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    focus_class: str,
    imgsz: int = 640,
    iou_threshold: float = 0.5,
    low_conf_threshold: float = 0.5,
) -> dict[str, Path]:
    """Analyze baseline predictions against YOLO labels and export research artifacts."""

    eval_dir = ensure_dir(eval_dir, "Evaluation directory")
    output_dir = ensure_dir(output_dir, "Error analysis output directory", create=True)
    config = load_yolo_dataset_config(data_path)
    metrics_path = ensure_file(eval_dir / "metrics_overall.json", "Overall metrics JSON")
    predictions_path = ensure_file(eval_dir / "prediction_rows.csv", "Prediction rows CSV")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    image_index = _build_image_index(config)
    allowed_splits = _infer_eval_splits(config, predictions_path, eval_dir / "chipped_error_cases.csv")
    gt_by_image, gt_counts = _load_ground_truth_boxes(config, imgsz, allowed_splits)
    pred_by_image = _load_prediction_boxes(predictions_path, image_index, config.names)

    image_keys = sorted(set(gt_by_image) | set(pred_by_image))
    class_names = list(config.names)
    summary_by_class = {class_name: _empty_class_summary(class_id, class_name) for class_id, class_name in enumerate(class_names)}
    focus_rows: list[dict[str, Any]] = []
    all_error_rows: list[dict[str, Any]] = []

    error_type_counts = Counter()
    error_tag_counts = Counter()
    focus_error_type_counts = Counter()
    focus_error_tag_counts = Counter()

    for class_name, count in gt_counts.items():
        summary_by_class[class_name]["gt_count"] = count

    for image_key in image_keys:
        gt_rows = gt_by_image.get(image_key, [])
        pred_rows = pred_by_image.get(image_key, [])
        matched_gt_indices, matched_pred_indices = _match_predictions(gt_rows, pred_rows, iou_threshold)

        for gt in gt_rows:
            class_summary = summary_by_class[gt.class_name]
            if gt.gt_index in matched_gt_indices:
                class_summary["tp"] += 1
                continue

            error_row = _build_gt_error_row(
                gt=gt,
                pred_rows=pred_rows,
                matched_pred_indices=matched_pred_indices,
                focus_class=focus_class,
                low_conf_threshold=low_conf_threshold,
                iou_threshold=iou_threshold,
            )
            all_error_rows.append(error_row)
            class_summary["fn"] += 1
            class_summary[error_row["primary_error_type"]] += 1
            for tag in error_row["error_tags"].split("|"):
                if tag:
                    class_summary[tag] += 1
                    error_tag_counts[tag] += 1
                    if gt.class_name == focus_class:
                        focus_error_tag_counts[tag] += 1
            error_type_counts[error_row["primary_error_type"]] += 1
            if gt.class_name == focus_class:
                focus_rows.append(error_row)
                focus_error_type_counts[error_row["primary_error_type"]] += 1

        for pred in pred_rows:
            if pred.pred_index in matched_pred_indices:
                continue
            error_row = _build_pred_error_row(
                pred=pred,
                gt_rows=gt_rows,
                matched_gt_indices=matched_gt_indices,
                focus_class=focus_class,
                low_conf_threshold=low_conf_threshold,
                iou_threshold=iou_threshold,
            )
            all_error_rows.append(error_row)
            class_summary = summary_by_class[pred.class_name]
            class_summary["fp"] += 1
            class_summary[error_row["primary_error_type"]] += 1
            for tag in error_row["error_tags"].split("|"):
                if tag:
                    class_summary[tag] += 1
                    error_tag_counts[tag] += 1
                    if pred.class_name == focus_class:
                        focus_error_tag_counts[tag] += 1
            error_type_counts[error_row["primary_error_type"]] += 1
            if pred.class_name == focus_class:
                focus_rows.append(error_row)
                focus_error_type_counts[error_row["primary_error_type"]] += 1

    _attach_metrics(summary_by_class, metrics)

    per_class_rows = _finalize_class_rows(summary_by_class)
    focus_summary = _build_focus_summary(
        focus_rows=focus_rows,
        focus_class=focus_class,
        metrics=metrics,
        summary_by_class=summary_by_class,
        eval_dir=eval_dir,
    )
    error_summary = _build_error_summary(
        eval_dir=eval_dir,
        data_path=Path(data_path).expanduser().resolve(),
        output_dir=output_dir,
        metrics=metrics,
        class_rows=per_class_rows,
        error_rows=all_error_rows,
        error_type_counts=error_type_counts,
        error_tag_counts=error_tag_counts,
        focus_class=focus_class,
        focus_rows=focus_rows,
    )
    size_rows = _build_size_bin_summary(all_error_rows)
    confidence_rows = _build_confidence_summary(all_error_rows)

    outputs = {
        "error_summary_json": _write_json(error_summary, output_dir / "error_summary.json"),
        "per_class_error_summary_csv": _write_csv(per_class_rows, output_dir / "per_class_error_summary.csv"),
        "focus_class_error_summary_json": _write_json(focus_summary, output_dir / "focus_class_error_summary.json"),
        "focus_class_error_cases_csv": _write_csv(focus_rows, output_dir / "focus_class_error_cases.csv"),
        "size_bin_error_summary_csv": _write_csv(size_rows, output_dir / "size_bin_error_summary.csv"),
        "confidence_error_summary_csv": _write_csv(confidence_rows, output_dir / "confidence_error_summary.csv"),
        "readme_md": _write_readme(
            output_dir=output_dir,
            focus_class=focus_class,
            metrics=metrics,
            focus_summary=focus_summary,
            has_plots=_collect_plot_availability(eval_dir),
        ),
    }
    return outputs


def _build_image_index(config) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for split in config.splits:
        for image_path in split.image_files:
            relative = image_path.relative_to(config.dataset_root).as_posix()
            index[relative] = image_path
            index[image_path.name] = image_path
    return index


def _infer_eval_splits(config, predictions_path: Path, chipped_csv_path: Path) -> set[str]:
    raw_paths: list[str] = []
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_paths.append(row["image_path"])
    if chipped_csv_path.exists():
        with chipped_csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_paths.append(row["image_path"])

    matched_splits: set[str] = set()
    for split in config.splits:
        relative_dir = split.image_dir.relative_to(config.dataset_root).as_posix()
        if any(relative_dir in raw_path.replace("\\", "/") for raw_path in raw_paths):
            matched_splits.add(split.name)

    if matched_splits:
        return matched_splits
    return {config.splits[-1].name}


def _load_ground_truth_boxes(config, imgsz: int, allowed_splits: set[str]) -> tuple[dict[str, list[GroundTruthBox]], Counter[str]]:
    gt_by_image: dict[str, list[GroundTruthBox]] = defaultdict(list)
    gt_counts: Counter[str] = Counter()

    for split in config.splits:
        if split.name not in allowed_splits:
            continue
        image_map = {path.stem: path for path in split.image_files}
        gt_index_by_image = Counter()
        for label_path in split.label_files:
            image_path = image_map.get(label_path.stem)
            if image_path is None:
                continue
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            width_px, height_px = Image.open(image_path).size
            image_key = image_path.relative_to(config.dataset_root).as_posix()
            for raw_line in text.splitlines():
                parts = raw_line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
                x1 = (x_center - box_width / 2.0) * width_px
                y1 = (y_center - box_height / 2.0) * height_px
                x2 = (x_center + box_width / 2.0) * width_px
                y2 = (y_center + box_height / 2.0) * height_px
                area_ratio = box_width * box_height
                gt_box = GroundTruthBox(
                    image_path=str(image_path),
                    split=split.name,
                    class_id=class_id,
                    class_name=config.names[class_id],
                    gt_index=int(gt_index_by_image[image_key]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    width=x2 - x1,
                    height=y2 - y1,
                    area_ratio=area_ratio,
                    aspect_ratio=box_width / box_height,
                    size_group=_assign_size_group(area_ratio),
                    pixel_width=box_width * imgsz,
                    pixel_height=box_height * imgsz,
                    pixel_area=(box_width * imgsz) * (box_height * imgsz),
                    pixel_size_group=_assign_pixel_size_group((box_width * imgsz) * (box_height * imgsz)),
                )
                gt_by_image[image_key].append(gt_box)
                gt_counts[gt_box.class_name] += 1
                gt_index_by_image[image_key] += 1

    return gt_by_image, gt_counts


def _load_prediction_boxes(path: Path, image_index: dict[str, Path], class_names: tuple[str, ...]) -> dict[str, list[PredictionBox]]:
    pred_by_image: dict[str, list[PredictionBox]] = defaultdict(list)
    pred_counter: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            resolved_image = _resolve_artifact_image_path(row["image_path"], image_index)
            image_key = resolved_image[1]
            class_id = int(row["class_id"])
            pred_box = PredictionBox(
                image_path=str(resolved_image[0]),
                class_id=class_id,
                class_name=class_names[class_id],
                pred_index=int(pred_counter[image_key]),
                confidence=float(row["confidence"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                width=max(0.0, float(row["x2"]) - float(row["x1"])),
                height=max(0.0, float(row["y2"]) - float(row["y1"])),
                area=max(0.0, float(row["x2"]) - float(row["x1"])) * max(0.0, float(row["y2"]) - float(row["y1"])),
            )
            pred_by_image[image_key].append(pred_box)
            pred_counter[image_key] += 1
    return pred_by_image


def _resolve_artifact_image_path(image_path: str, image_index: dict[str, Path]) -> tuple[Path, str]:
    path_obj = Path(image_path)
    candidates = []
    if "images" in path_obj.parts:
        image_idx = path_obj.parts.index("images")
        candidates.append("/".join(path_obj.parts[image_idx:]))
    candidates.append(path_obj.name)
    for candidate in candidates:
        resolved = image_index.get(candidate)
        if resolved is not None:
            return resolved, resolved.relative_to(resolved.parents[2]).as_posix()
    if path_obj.exists():
        return path_obj.resolve(), path_obj.name
    raise FileNotFoundError(f"Could not map artifact image path to a local dataset image: {image_path}")


def _match_predictions(
    gt_rows: list[GroundTruthBox],
    pred_rows: list[PredictionBox],
    iou_threshold: float,
) -> tuple[set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt in gt_rows:
        for pred in pred_rows:
            if gt.class_id != pred.class_id:
                continue
            iou = _xyxy_iou((gt.x1, gt.y1, gt.x2, gt.y2), (pred.x1, pred.y1, pred.x2, pred.y2))
            if iou >= iou_threshold:
                candidates.append((iou, gt.gt_index, pred.pred_index))

    candidates.sort(reverse=True)
    matched_gt_indices: set[int] = set()
    matched_pred_indices: set[int] = set()
    for _, gt_index, pred_index in candidates:
        if gt_index in matched_gt_indices or pred_index in matched_pred_indices:
            continue
        matched_gt_indices.add(gt_index)
        matched_pred_indices.add(pred_index)
    return matched_gt_indices, matched_pred_indices


def _build_gt_error_row(
    gt: GroundTruthBox,
    pred_rows: list[PredictionBox],
    matched_pred_indices: set[int],
    focus_class: str,
    low_conf_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    best_same = None
    best_same_iou = 0.0
    best_any = None
    best_any_iou = 0.0

    for pred in pred_rows:
        if pred.pred_index in matched_pred_indices:
            continue
        iou = _xyxy_iou((gt.x1, gt.y1, gt.x2, gt.y2), (pred.x1, pred.y1, pred.x2, pred.y2))
        if iou > best_any_iou:
            best_any_iou = iou
            best_any = pred
        if pred.class_id == gt.class_id and iou > best_same_iou:
            best_same_iou = iou
            best_same = pred

    tags = ["false_negative"]
    primary = "false_negative"
    notes: list[str] = ["Ground-truth object was not matched by a same-class prediction at IoU >= threshold."]
    if best_same is not None and best_same_iou >= 0.1:
        primary = "localization_error"
        tags.append("localization_error")
        notes.append(f"Best same-class prediction reached IoU {best_same_iou:.3f} but stayed below {iou_threshold:.2f}.")
        if best_same.confidence < low_conf_threshold:
            tags.append("low_confidence_detection")
            notes.append(f"Best same-class prediction confidence {best_same.confidence:.3f} was below {low_conf_threshold:.2f}.")
    if best_any is not None and best_any.class_id != gt.class_id and best_any_iou >= 0.1:
        tags.append("class_confusion")
        notes.append(f"Best overlapping prediction was class {best_any.class_name} with IoU {best_any_iou:.3f}.")
    if gt.size_group in {"tiny", "small"} or gt.pixel_size_group in {"tiny_px", "small_px"}:
        tags.append("small_or_weak_evidence_case")
    if (best_same is not None and 0.4 <= best_same_iou < iou_threshold) or (best_any is not None and best_any_iou >= 0.4):
        tags.append("ambiguous_or_possible_label_noise")
        notes.append("Near-threshold overlap suggests a review-worthy borderline case.")

    used_pred = best_same if best_same is not None else best_any
    used_iou = best_same_iou if best_same is not None else best_any_iou

    return {
        "image_path": gt.image_path,
        "split": gt.split,
        "focus_class": focus_class,
        "error_scope": "focus_class" if gt.class_name == focus_class else "other_class",
        "primary_error_type": primary,
        "error_tags": "|".join(dict.fromkeys(tags)),
        "source_role": "ground_truth",
        "gt_class_name": gt.class_name,
        "pred_class_name": "" if used_pred is None else used_pred.class_name,
        "gt_index": gt.gt_index,
        "pred_index": "" if used_pred is None else used_pred.pred_index,
        "best_iou": round(used_iou, 6),
        "confidence": "" if used_pred is None else round(used_pred.confidence, 6),
        "size_group": gt.size_group,
        "pixel_size_group": gt.pixel_size_group,
        "pixel_area": round(gt.pixel_area, 2),
        "gt_x1": round(gt.x1, 3),
        "gt_y1": round(gt.y1, 3),
        "gt_x2": round(gt.x2, 3),
        "gt_y2": round(gt.y2, 3),
        "pred_x1": "" if used_pred is None else round(used_pred.x1, 3),
        "pred_y1": "" if used_pred is None else round(used_pred.y1, 3),
        "pred_x2": "" if used_pred is None else round(used_pred.x2, 3),
        "pred_y2": "" if used_pred is None else round(used_pred.y2, 3),
        "notes": " ".join(notes),
    }


def _build_pred_error_row(
    pred: PredictionBox,
    gt_rows: list[GroundTruthBox],
    matched_gt_indices: set[int],
    focus_class: str,
    low_conf_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    best_same = None
    best_same_iou = 0.0
    best_any = None
    best_any_iou = 0.0

    for gt in gt_rows:
        if gt.gt_index in matched_gt_indices:
            continue
        iou = _xyxy_iou((gt.x1, gt.y1, gt.x2, gt.y2), (pred.x1, pred.y1, pred.x2, pred.y2))
        if iou > best_any_iou:
            best_any_iou = iou
            best_any = gt
        if gt.class_id == pred.class_id and iou > best_same_iou:
            best_same_iou = iou
            best_same = gt

    tags = ["false_positive"]
    primary = "false_positive"
    notes = ["Prediction did not match any same-class ground truth at IoU >= threshold."]
    if best_same is not None and best_same_iou >= 0.1:
        tags.append("localization_error")
        notes.append(f"Prediction overlapped a same-class object at IoU {best_same_iou:.3f} but stayed below {iou_threshold:.2f}.")
    if best_any is not None and best_any.class_id != pred.class_id and best_any_iou >= 0.1:
        tags.append("class_confusion")
        notes.append(f"Prediction overlapped a ground-truth object of class {best_any.class_name} at IoU {best_any_iou:.3f}.")
    if pred.confidence < low_conf_threshold:
        tags.append("low_confidence_detection")
        notes.append(f"Prediction confidence {pred.confidence:.3f} was below {low_conf_threshold:.2f}.")
    if best_any is not None and (best_any.size_group in {"tiny", "small"} or best_any.pixel_size_group in {"tiny_px", "small_px"}):
        tags.append("small_or_weak_evidence_case")
    if (best_same is not None and 0.4 <= best_same_iou < iou_threshold) or (best_any is not None and best_any_iou >= 0.4):
        tags.append("ambiguous_or_possible_label_noise")
        notes.append("Near-threshold overlap suggests a review-worthy borderline case.")

    return {
        "image_path": pred.image_path,
        "split": "" if best_any is None else best_any.split,
        "focus_class": focus_class,
        "error_scope": "focus_class" if pred.class_name == focus_class else "other_class",
        "primary_error_type": primary,
        "error_tags": "|".join(dict.fromkeys(tags)),
        "source_role": "prediction",
        "gt_class_name": "" if best_any is None else best_any.class_name,
        "pred_class_name": pred.class_name,
        "gt_index": "" if best_any is None else best_any.gt_index,
        "pred_index": pred.pred_index,
        "best_iou": round(best_same_iou if best_same is not None else best_any_iou, 6),
        "confidence": round(pred.confidence, 6),
        "size_group": "" if best_any is None else best_any.size_group,
        "pixel_size_group": "" if best_any is None else best_any.pixel_size_group,
        "pixel_area": "" if best_any is None else round(best_any.pixel_area, 2),
        "gt_x1": "" if best_any is None else round(best_any.x1, 3),
        "gt_y1": "" if best_any is None else round(best_any.y1, 3),
        "gt_x2": "" if best_any is None else round(best_any.x2, 3),
        "gt_y2": "" if best_any is None else round(best_any.y2, 3),
        "pred_x1": round(pred.x1, 3),
        "pred_y1": round(pred.y1, 3),
        "pred_x2": round(pred.x2, 3),
        "pred_y2": round(pred.y2, 3),
        "notes": " ".join(notes),
    }


def _attach_metrics(summary_by_class: dict[str, dict[str, Any]], metrics: dict[str, Any]) -> None:
    metric_lookup = {row["class_name"]: row["map"] for row in metrics.get("per_class", [])}
    for class_name, row in summary_by_class.items():
        row["ap50_95"] = metric_lookup.get(class_name)


def _finalize_class_rows(summary_by_class: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_name in sorted(summary_by_class, key=lambda name: summary_by_class[name]["class_id"]):
        row = dict(summary_by_class[class_name])
        gt_count = row["gt_count"]
        tp = row["tp"]
        fp = row["fp"]
        row["recall_proxy"] = round(tp / gt_count, 6) if gt_count else None
        row["precision_proxy"] = round(tp / (tp + fp), 6) if (tp + fp) else None
        rows.append(row)
    return rows


def _build_focus_summary(
    focus_rows: list[dict[str, Any]],
    focus_class: str,
    metrics: dict[str, Any],
    summary_by_class: dict[str, dict[str, Any]],
    eval_dir: Path,
) -> dict[str, Any]:
    class_row = summary_by_class.get(focus_class, {})
    tag_counts = Counter()
    primary_counts = Counter()
    zero_iou = 0
    near_threshold = 0
    with_overlap = 0
    for row in focus_rows:
        primary_counts[row["primary_error_type"]] += 1
        if float(row["best_iou"] or 0.0) == 0.0:
            zero_iou += 1
        if float(row["best_iou"] or 0.0) >= 0.4:
            near_threshold += 1
        if float(row["best_iou"] or 0.0) > 0.0:
            with_overlap += 1
        for tag in str(row["error_tags"]).split("|"):
            if tag:
                tag_counts[tag] += 1

    per_class_metrics = next((row for row in metrics.get("per_class", []) if row.get("class_name") == focus_class), {})
    limitations = _build_limitations(eval_dir)
    return {
        "focus_class": focus_class,
        "metrics": {
            "ap50_95": per_class_metrics.get("map"),
            "precision_proxy": None if not class_row else (round(class_row["tp"] / (class_row["tp"] + class_row["fp"]), 6) if (class_row["tp"] + class_row["fp"]) else None),
            "recall_proxy": None if not class_row else (round(class_row["tp"] / class_row["gt_count"], 6) if class_row["gt_count"] else None),
        },
        "counts": {
            "gt_count": class_row.get("gt_count", 0),
            "tp": class_row.get("tp", 0),
            "fn": class_row.get("fn", 0),
            "fp": class_row.get("fp", 0),
            "focus_error_rows": len(focus_rows),
            "zero_iou_errors": zero_iou,
            "nonzero_iou_errors": with_overlap,
            "near_threshold_overlap_errors": near_threshold,
        },
        "primary_error_type_counts": dict(primary_counts),
        "error_tag_counts": dict(tag_counts),
        "limitations": limitations,
        "interpretation": _build_focus_interpretation(focus_class, class_row, primary_counts, tag_counts, zero_iou, near_threshold),
    }


def _build_error_summary(
    eval_dir: Path,
    data_path: Path,
    output_dir: Path,
    metrics: dict[str, Any],
    class_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    error_type_counts: Counter[str],
    error_tag_counts: Counter[str],
    focus_class: str,
    focus_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    limitations = _build_limitations(eval_dir)
    return {
        "eval_dir": str(eval_dir.resolve()),
        "dataset_config": str(data_path),
        "output_dir": str(output_dir.resolve()),
        "overall_metrics": metrics.get("overall", {}),
        "focus_class": focus_class,
        "class_count": len(class_rows),
        "error_row_count": len(error_rows),
        "focus_error_row_count": len(focus_rows),
        "primary_error_type_counts": dict(error_type_counts),
        "error_tag_counts": dict(error_tag_counts),
        "available_plots": _collect_plot_availability(eval_dir),
        "limitations": limitations,
        "todo_for_phase5": [
            "Export per-ground-truth match metadata directly from evaluation so near-threshold cases do not rely on post-hoc heuristics.",
            "Save richer false-positive metadata, including NMS-suppressed candidates if future analysis needs confidence dynamics.",
            "Add optional saliency/evidence placeholders only after the Phase 5 XAI hook design is implemented.",
        ],
    }


def _build_size_bin_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = Counter()
    for row in rows:
        normalized = row.get("size_group") or "unknown"
        pixel = row.get("pixel_size_group") or "unknown"
        primary = row["primary_error_type"]
        summary[(normalized, pixel, primary)] += 1
    records = []
    for (normalized, pixel, primary), count in sorted(summary.items()):
        records.append(
            {
                "size_group": normalized,
                "pixel_size_group": pixel,
                "primary_error_type": primary,
                "count": count,
            }
        )
    return records


def _build_confidence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [
        ("missing", None, None),
        ("0.00-0.25", 0.0, 0.25),
        ("0.25-0.50", 0.25, 0.5),
        ("0.50-0.75", 0.5, 0.75),
        ("0.75-1.00", 0.75, 1.000001),
    ]
    counts = Counter()
    for row in rows:
        confidence = row.get("confidence")
        primary = row["primary_error_type"]
        if confidence in {"", None}:
            counts[("missing", primary)] += 1
            continue
        value = float(confidence)
        for label, low, high in bins[1:]:
            if low <= value < high:
                counts[(label, primary)] += 1
                break
    records = []
    for (label, primary), count in sorted(counts.items()):
        records.append(
            {
                "confidence_bin": label,
                "primary_error_type": primary,
                "count": count,
            }
        )
    return records


def _write_readme(
    output_dir: Path,
    focus_class: str,
    metrics: dict[str, Any],
    focus_summary: dict[str, Any],
    has_plots: dict[str, bool],
) -> Path:
    overall = metrics.get("overall", {})
    lines = [
        "# Baseline Error Analysis",
        "",
        "This directory contains heuristic post-hoc error analysis built from baseline evaluation artifacts and YOLO labels.",
        "",
        "## Inputs",
        "",
        f"- Focus class: `{focus_class}`",
        f"- Overall mAP50-95: `{overall.get('map')}`",
        f"- Overall mAP50: `{overall.get('map50')}`",
        f"- Overall mAP75: `{overall.get('map75')}`",
        "",
        "## Output Guide",
        "",
        "- `error_summary.json`: high-level summary, limitations, available plots, and Phase 5 TODOs.",
        "- `per_class_error_summary.csv`: per-class TP/FN/FP proxies plus heuristic error counts.",
        "- `focus_class_error_summary.json`: focused summary for the chosen class.",
        "- `focus_class_error_cases.csv`: row-level error cases for the focus class with GT/pred boxes when available.",
        "- `size_bin_error_summary.csv`: error counts grouped by normalized and projected pixel size bins.",
        "- `confidence_error_summary.csv`: error counts grouped by confidence bins when confidence exists.",
        "",
        "## Plot Availability",
        "",
        f"- Confusion matrix: `{has_plots.get('confusion_matrix', False)}`",
        f"- PR curve: `{has_plots.get('pr_curve', False)}`",
        f"- F1 curve: `{has_plots.get('f1_curve', False)}`",
        "",
        "## Manual Review Notes",
        "",
        "- Treat `localization_error`, `class_confusion`, and `ambiguous_or_possible_label_noise` as heuristic tags from post-hoc box overlap, not as causal conclusions.",
        "- For the focus class, prioritize cases with non-zero IoU but failed matches because they are stronger candidates for future evidence-map inspection.",
        "- Zero-IoU false negatives still matter because they may indicate weak evidence, missed visual cues, or annotation ambiguity, but they cannot yet be separated without Phase 5 hooks.",
        "",
        "## Focus-Class Snapshot",
        "",
        f"- Focus rows: `{focus_summary['counts']['focus_error_rows']}`",
        f"- False negatives: `{focus_summary['counts']['fn']}`",
        f"- False positives: `{focus_summary['counts']['fp']}`",
        f"- Near-threshold overlap errors: `{focus_summary['counts']['near_threshold_overlap_errors']}`",
    ]
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def _collect_plot_availability(eval_dir: Path) -> dict[str, bool]:
    val_dir = eval_dir / "val"
    return {
        "confusion_matrix": (val_dir / "confusion_matrix.png").exists(),
        "confusion_matrix_normalized": (val_dir / "confusion_matrix_normalized.png").exists(),
        "pr_curve": (val_dir / "BoxPR_curve.png").exists(),
        "f1_curve": (val_dir / "BoxF1_curve.png").exists(),
        "precision_curve": (val_dir / "BoxP_curve.png").exists(),
        "recall_curve": (val_dir / "BoxR_curve.png").exists(),
    }


def _build_limitations(eval_dir: Path) -> list[str]:
    limitations = [
        "The analysis uses post-hoc IoU matching over exported prediction rows rather than internal validator match assignments.",
        "Current artifacts do not include saliency maps, feature activations, or explanation signals, so no evidence-based claim can be made yet.",
        "Class-confusion and annotation-ambiguity labels are heuristic tags derived from overlap patterns, not manual adjudication.",
    ]
    if not (eval_dir / "val" / "predictions.json").exists():
        limitations.append("COCO-style predictions.json was not available, limiting some downstream compatibility checks.")
    return limitations


def _build_focus_interpretation(
    focus_class: str,
    class_row: dict[str, Any],
    primary_counts: Counter[str],
    tag_counts: Counter[str],
    zero_iou: int,
    near_threshold: int,
) -> list[str]:
    findings = []
    if class_row:
        findings.append(
            f"{focus_class} remains a hard class: AP50-95 is {class_row.get('ap50_95')}, with TP/FN/FP proxies {class_row.get('tp')}/{class_row.get('fn')}/{class_row.get('fp')}."
        )
    if zero_iou:
        findings.append(f"{zero_iou} focus-class errors had zero overlap with the best retained prediction, suggesting many outright misses.")
    if near_threshold:
        findings.append(f"{near_threshold} focus-class errors were near the IoU threshold, making them strong candidates for later evidence-map review.")
    if tag_counts.get("small_or_weak_evidence_case"):
        findings.append(
            f"{tag_counts['small_or_weak_evidence_case']} focus-class errors fall into small-object or weak-evidence heuristics, consistent with the repo's current framing."
        )
    if tag_counts.get("ambiguous_or_possible_label_noise"):
        findings.append("Some focus-class cases look borderline enough to justify annotation review before stronger causal claims.")
    if not findings:
        findings.append(f"{focus_class} did not produce enough tagged errors for a meaningful focused interpretation.")
    return findings


def _empty_class_summary(class_id: int, class_name: str) -> dict[str, Any]:
    return {
        "class_id": class_id,
        "class_name": class_name,
        "gt_count": 0,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "false_negative": 0,
        "false_positive": 0,
        "localization_error": 0,
        "class_confusion": 0,
        "low_confidence_detection": 0,
        "small_or_weak_evidence_case": 0,
        "ambiguous_or_possible_label_noise": 0,
        "ap50_95": None,
    }


def _write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["note"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return output_path


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
