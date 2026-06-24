"""Simple post-hoc evidence metrics for CAM maps."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_evidence_metrics(
    cam_map: np.ndarray,
    gt_box: tuple[float, float, float, float] | None,
    pred_box: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    """Compute lightweight evidence statistics on a normalized CAM map."""

    total_energy = float(cam_map.sum())
    peak_y, peak_x = np.unravel_index(int(cam_map.argmax()), cam_map.shape)
    metrics = {
        "cam_total_energy": total_energy,
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "peak_value": float(cam_map[peak_y, peak_x]),
        "saliency_concentration": _saliency_concentration(cam_map),
    }

    gt_energy, peak_inside_gt = _box_energy_and_peak(cam_map, gt_box, peak_x, peak_y, total_energy)
    pred_energy, _ = _box_energy_and_peak(cam_map, pred_box, peak_x, peak_y, total_energy)
    metrics["energy_in_gt_box"] = gt_energy
    metrics["energy_in_pred_box"] = pred_energy
    metrics["peak_inside_gt_box"] = peak_inside_gt
    return metrics


def _saliency_concentration(cam_map: np.ndarray, top_fraction: float = 0.05) -> float:
    flat = np.sort(cam_map.reshape(-1))
    if flat.size == 0:
        return 0.0
    top_k = max(1, int(math.ceil(flat.size * top_fraction)))
    total = float(flat.sum())
    if total <= 0.0:
        return 0.0
    return float(flat[-top_k:].sum() / total)


def _box_energy_and_peak(
    cam_map: np.ndarray,
    box: tuple[float, float, float, float] | None,
    peak_x: int,
    peak_y: int,
    total_energy: float,
) -> tuple[float | None, bool | None]:
    if box is None:
        return None, None
    x1, y1, x2, y2 = _clamp_box(box, cam_map.shape[1], cam_map.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0, False
    region = cam_map[y1:y2, x1:x2]
    energy = 0.0 if total_energy <= 0.0 else float(region.sum() / total_energy)
    peak_inside = x1 <= peak_x < x2 and y1 <= peak_y < y2
    return energy, peak_inside


def _clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )
