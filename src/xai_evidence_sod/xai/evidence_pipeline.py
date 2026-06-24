"""Phase 5 post-hoc XAI evidence extraction pipeline."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

from .cam import build_cam_extractor, preprocess_image_for_yolo
from .case_selection import CuratedCase, build_curated_cases, summarize_case_groups
from .evidence_metrics import compute_evidence_metrics


def run_xai_evidence_extraction(
    weights_path: str | Path,
    data_path: str | Path,
    cases_csv: str | Path,
    output_dir: str | Path,
    focus_class: str,
    methods: list[str],
    max_cases: int | None,
    seed: int,
    imgsz: int = 640,
    device: str = "cpu",
    target_layer: int = 18,
    prediction_rows_csv: str | Path = "artifacts/baseline_eval/prediction_rows.csv",
) -> dict[str, Path]:
    """Extract post-hoc XAI evidence artifacts on curated focus-class cases."""

    output_dir = Path(output_dir).expanduser().resolve()
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "crops"
    sheets_dir = output_dir / "contact_sheets"
    maps_dir = output_dir / "maps"
    for directory in (output_dir, overlays_dir, crops_dir, sheets_dir, maps_dir):
        directory.mkdir(parents=True, exist_ok=True)

    curated_cases = build_curated_cases(
        cases_csv=cases_csv,
        data_path=data_path,
        prediction_rows_csv=prediction_rows_csv,
        focus_class=focus_class,
        max_cases=max_cases,
        seed=seed,
        imgsz=imgsz,
    )

    yolo = YOLO(str(Path(weights_path).expanduser().resolve()))
    yolo.model.to(device)

    evidence_rows: list[dict[str, Any]] = []
    sheet_index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for method in methods:
        extractor = build_cam_extractor(method=method, model=yolo.model, target_layer=target_layer)
        for case in curated_cases:
            prepared = preprocess_image_for_yolo(case.image_path, imgsz=imgsz, device=device)
            cam_result = extractor.extract(prepared)
            metrics = compute_evidence_metrics(
                cam_map=cam_result.cam_original,
                gt_box=_case_box(case, "gt"),
                pred_box=_case_box(case, "pred"),
            )
            overlay_path, crop_path, map_path = _export_case_visuals(
                case=case,
                cam_map=cam_result.cam_original,
                method=method,
                prepared=prepared,
                overlays_dir=overlays_dir,
                crops_dir=crops_dir,
                maps_dir=maps_dir,
            )
            sheet_index[(method, case.case_group)].append(overlay_path)
            evidence_rows.append(
                {
                    "case_id": case.case_id,
                    "method": method,
                    "target_layer": target_layer,
                    "image_path": case.image_path,
                    "split": case.split,
                    "focus_class": case.focus_class,
                    "case_group": case.case_group,
                    "source_role": case.source_role,
                    "gt_class_name": case.gt_class_name,
                    "pred_class_name": case.pred_class_name,
                    "best_iou": round(case.best_iou, 6),
                    "confidence": None if case.confidence is None else round(case.confidence, 6),
                    "is_near_threshold": case.is_near_threshold,
                    "size_group": case.size_group,
                    "pixel_size_group": case.pixel_size_group,
                    "pixel_area": case.pixel_area,
                    "energy_in_gt_box": metrics["energy_in_gt_box"],
                    "energy_in_pred_box": metrics["energy_in_pred_box"],
                    "peak_inside_gt_box": metrics["peak_inside_gt_box"],
                    "saliency_concentration": round(metrics["saliency_concentration"], 6),
                    "peak_x": metrics["peak_x"],
                    "peak_y": metrics["peak_y"],
                    "peak_value": round(metrics["peak_value"], 6),
                    "overlay_path": str(overlay_path),
                    "crop_path": str(crop_path),
                    "map_path": str(map_path),
                    "notes": case.notes,
                }
            )

    contact_sheet_paths = _build_contact_sheets(sheet_index, sheets_dir)
    evidence_csv = _write_csv(evidence_rows, output_dir / "evidence_cases.csv")
    summary_json = _write_json(
        _build_summary(
            curated_cases=curated_cases,
            evidence_rows=evidence_rows,
            focus_class=focus_class,
            methods=methods,
            target_layer=target_layer,
            max_cases=max_cases,
            seed=seed,
            contact_sheet_paths=contact_sheet_paths,
        ),
        output_dir / "evidence_summary.json",
    )
    readme_md = _write_readme(
        output_dir=output_dir,
        focus_class=focus_class,
        methods=methods,
        target_layer=target_layer,
        evidence_rows=evidence_rows,
        contact_sheet_paths=contact_sheet_paths,
    )
    return {
        "evidence_cases_csv": evidence_csv,
        "evidence_summary_json": summary_json,
        "readme_md": readme_md,
    }


def _export_case_visuals(
    case: CuratedCase,
    cam_map: np.ndarray,
    method: str,
    prepared,
    overlays_dir: Path,
    crops_dir: Path,
    maps_dir: Path,
) -> tuple[Path, Path, Path]:
    overlay = _make_overlay(prepared.original_rgb, cam_map, case)
    crop = _make_crop(prepared.original_rgb, cam_map, case)
    heatmap = Image.fromarray((_heatmap_rgb(cam_map) * 255).astype(np.uint8))

    stem = f"{method}__{case.case_group}__{case.case_id}__{Path(case.image_path).stem}"
    overlay_path = overlays_dir / f"{stem}.jpg"
    crop_path = crops_dir / f"{stem}.jpg"
    map_path = maps_dir / f"{stem}.png"
    overlay.save(overlay_path, quality=90)
    crop.save(crop_path, quality=90)
    heatmap.save(map_path)
    return overlay_path, crop_path, map_path


def _make_overlay(original_rgb: np.ndarray, cam_map: np.ndarray, case: CuratedCase) -> Image.Image:
    image = Image.fromarray(original_rgb).convert("RGB")
    heatmap = (_heatmap_rgb(cam_map) * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap).convert("RGB")
    blended = Image.blend(image, heatmap_image, alpha=0.45)
    draw = ImageDraw.Draw(blended)
    gt_box = _case_box(case, "gt")
    pred_box = _case_box(case, "pred")
    if gt_box is not None:
        draw.rectangle(gt_box, outline=(0, 220, 80), width=4)
    if pred_box is not None:
        draw.rectangle(pred_box, outline=(255, 140, 0), width=4)
    draw.rectangle((0, 0, blended.width, 30), fill=(20, 20, 20))
    draw.text((8, 8), f"{case.case_group} | iou={case.best_iou:.3f}", fill=(255, 255, 255))
    return blended


def _make_crop(original_rgb: np.ndarray, cam_map: np.ndarray, case: CuratedCase) -> Image.Image:
    base = _make_overlay(original_rgb, cam_map, case)
    reference = _case_box(case, "gt") or _case_box(case, "pred")
    if reference is None:
        return base
    margin = 48
    x1 = max(0, int(reference[0]) - margin)
    y1 = max(0, int(reference[1]) - margin)
    x2 = min(base.width, int(reference[2]) + margin)
    y2 = min(base.height, int(reference[3]) + margin)
    return base.crop((x1, y1, x2, y2))


def _build_contact_sheets(sheet_index: dict[tuple[str, str], list[Path]], sheets_dir: Path) -> list[str]:
    outputs: list[str] = []
    for (method, group), paths in sorted(sheet_index.items()):
        if not paths:
            continue
        sheet_path = sheets_dir / f"{method}__{group}.jpg"
        _make_contact_sheet(paths, sheet_path)
        outputs.append(str(sheet_path))
    return outputs


def _make_contact_sheet(paths: list[Path], output_path: Path, thumb_size: tuple[int, int] = (256, 256), columns: int = 4) -> None:
    tiles = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size)
        tile = Image.new("RGB", thumb_size, color=(245, 245, 245))
        x_offset = (thumb_size[0] - image.width) // 2
        y_offset = (thumb_size[1] - image.height) // 2
        tile.paste(image, (x_offset, y_offset))
        tile = ImageOps.expand(tile, border=2, fill=(25, 25, 25))
        tiles.append(tile)
    rows_needed = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (thumb_size[0] + 4), rows_needed * (thumb_size[1] + 4)), color=(250, 250, 250))
    for index, tile in enumerate(tiles):
        col = index % columns
        row = index // columns
        sheet.paste(tile, (col * (thumb_size[0] + 4), row * (thumb_size[1] + 4)))
    sheet.save(output_path, quality=90)


def _build_summary(
    curated_cases: list[CuratedCase],
    evidence_rows: list[dict[str, Any]],
    focus_class: str,
    methods: list[str],
    target_layer: int,
    max_cases: int | None,
    seed: int,
    contact_sheet_paths: list[str],
) -> dict[str, Any]:
    group_counts = summarize_case_groups(curated_cases)
    method_counts = Counter(row["method"] for row in evidence_rows)
    concentration_values = [float(row["saliency_concentration"]) for row in evidence_rows]
    gt_energy_values = [float(row["energy_in_gt_box"]) for row in evidence_rows if row["energy_in_gt_box"] is not None]
    pred_energy_values = [float(row["energy_in_pred_box"]) for row in evidence_rows if row["energy_in_pred_box"] is not None]

    return {
        "focus_class": focus_class,
        "methods": methods,
        "target_layer": target_layer,
        "requested_max_cases": max_cases,
        "selected_case_count": len(curated_cases),
        "seed": seed,
        "group_counts": group_counts,
        "method_counts": dict(method_counts),
        "average_saliency_concentration": None if not concentration_values else round(sum(concentration_values) / len(concentration_values), 6),
        "average_energy_in_gt_box": None if not gt_energy_values else round(sum(gt_energy_values) / len(gt_energy_values), 6),
        "average_energy_in_pred_box": None if not pred_energy_values else round(sum(pred_energy_values) / len(pred_energy_values), 6),
        "contact_sheets": contact_sheet_paths,
        "limitations": [
            "Phase 5 implements EigenCAM first; Grad-CAM variants are only scaffolded in the interface.",
            "The current evidence metrics are descriptive post-hoc summaries, not proof that the detector uses the highlighted regions causally.",
            "True positives are proxy matches reconstructed from exported predictions and test labels.",
        ],
    }


def _write_readme(
    output_dir: Path,
    focus_class: str,
    methods: list[str],
    target_layer: int,
    evidence_rows: list[dict[str, Any]],
    contact_sheet_paths: list[str],
) -> Path:
    group_counts = Counter(row["case_group"] for row in evidence_rows)
    lines = [
        "# XAI Evidence Extraction",
        "",
        f"- Focus class: `{focus_class}`",
        f"- Methods: `{', '.join(methods)}`",
        f"- Target layer index: `{target_layer}`",
        f"- Evidence rows: `{len(evidence_rows)}`",
        "",
        "## Case Groups",
        "",
    ]
    for group, count in sorted(group_counts.items()):
        lines.append(f"- `{group}`: `{count}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "- `energy_in_gt_box`: fraction of CAM energy inside the GT box.",
            "- `energy_in_pred_box`: fraction of CAM energy inside the predicted box when one exists.",
            "- `peak_inside_gt_box`: whether the max-saliency pixel falls inside the GT box.",
            "- `saliency_concentration`: fraction of total energy contained in the top 5 percent of CAM pixels.",
            "",
            "## Review Guidance",
            "",
            "- Compare `true_positive_proxy` against `false_negative` and `localization_error` cases first.",
            "- Use near-threshold and class-confusion cases to judge whether the evidence map aligns with subtle defect cues or nearby distractors.",
            "- Do not treat these overlays as proof of causality or as evidence that XAI already improves the model.",
            "",
            "## Contact Sheets",
            "",
        ]
    )
    for path in contact_sheet_paths:
        lines.append(f"- `{path}`")
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    fieldnames = list(rows[0].keys()) if rows else ["note"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return output_path


def _write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _case_box(case: CuratedCase, prefix: str) -> tuple[float, float, float, float] | None:
    values = (
        getattr(case, f"{prefix}_x1"),
        getattr(case, f"{prefix}_y1"),
        getattr(case, f"{prefix}_x2"),
        getattr(case, f"{prefix}_y2"),
    )
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)


def _heatmap_rgb(cam_map: np.ndarray) -> np.ndarray:
    """Map a normalized CAM to a simple yellow-red heatmap."""

    cam = np.clip(cam_map.astype(np.float32), 0.0, 1.0)
    red = cam
    green = np.sqrt(cam)
    blue = np.power(cam, 3.0) * 0.15
    return np.stack([red, green, blue], axis=-1)
