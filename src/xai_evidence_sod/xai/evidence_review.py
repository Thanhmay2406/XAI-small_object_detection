"""Phase 6 review utilities for post-hoc XAI evidence artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


REQUIRED_GROUPS = [
    "true_positive_proxy",
    "false_negative",
    "false_positive",
    "localization_error",
    "near_threshold_overlap",
]
OPTIONAL_METRIC_COLUMNS = [
    "energy_in_gt_box",
    "energy_in_pred_box",
    "peak_inside_gt_box",
    "saliency_concentration",
]


def review_xai_evidence(
    evidence_csv: str | Path,
    output_dir: str | Path,
    focus_class: str | None = None,
    top_k: int = 8,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Build Phase 6 review artifacts from Phase 5 evidence CSV."""

    evidence_csv = Path(evidence_csv)
    output_dir = Path(output_dir)
    project_root_path = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")

    rows, warnings = _load_rows(evidence_csv=evidence_csv, focus_class=focus_class, project_root=project_root_path)
    if not rows:
        raise ValueError(f"No evidence rows available after filtering: {evidence_csv}")

    group_summary_rows = build_group_summary_rows(rows)
    representative_rows = select_representative_cases(rows, top_k=top_k)

    summary_payload = {
        "focus_class": focus_class or "all",
        "source_evidence_csv": _to_repo_relative(evidence_csv.resolve(), project_root_path),
        "row_count": len(rows),
        "top_k": top_k,
        "warnings": warnings,
        "metric_columns_present": {
            column: any(row.get(column) is not None for row in rows)
            for column in OPTIONAL_METRIC_COLUMNS
        },
        "group_summary": group_summary_rows,
        "representative_case_counts": _count_by_key(representative_rows, "representative_group"),
        "representative_cases_csv": "representative_cases.csv",
        "review_notes_template_csv": "review_notes_template.csv",
    }

    summary_csv = _write_csv(group_summary_rows, output_dir / "evidence_group_summary.csv")
    summary_json = _write_json(summary_payload, output_dir / "evidence_group_summary.json")
    representatives_csv = _write_csv(representative_rows, output_dir / "representative_cases.csv")
    notes_csv = _write_csv(_build_review_notes_template(representative_rows), output_dir / "review_notes_template.csv")
    readme_md = _write_readme(
        output_dir=output_dir,
        focus_class=focus_class or "all",
        top_k=top_k,
        warnings=warnings,
        group_summary_rows=group_summary_rows,
        representative_rows=representative_rows,
    )
    plot_paths = _maybe_write_plots(output_dir=output_dir, group_summary_rows=group_summary_rows, warnings=warnings)
    if plot_paths:
        summary_payload["plots"] = [path.name for path in plot_paths]
        _write_json(summary_payload, output_dir / "evidence_group_summary.json")

    return {
        "evidence_group_summary_csv": summary_csv,
        "evidence_group_summary_json": summary_json,
        "representative_cases_csv": representatives_csv,
        "review_notes_template_csv": notes_csv,
        "readme_md": readme_md,
        **{f"plot_{index + 1}": path for index, path in enumerate(plot_paths)},
    }


def build_group_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute descriptive statistics for each review group."""

    grouped_rows = {
        "true_positive_proxy": [row for row in rows if row.get("case_group") == "true_positive_proxy"],
        "false_negative": [row for row in rows if row.get("case_group") == "false_negative"],
        "false_positive": [row for row in rows if row.get("case_group") == "false_positive"],
        "localization_error": [row for row in rows if row.get("case_group") == "localization_error"],
        "near_threshold_overlap": [row for row in rows if row.get("is_near_threshold") is True],
    }

    summary_rows: list[dict[str, Any]] = []
    for group_name in REQUIRED_GROUPS:
        group_rows = grouped_rows[group_name]
        gt_values = [row["energy_in_gt_box"] for row in group_rows if row.get("energy_in_gt_box") is not None]
        pred_values = [row["energy_in_pred_box"] for row in group_rows if row.get("energy_in_pred_box") is not None]
        saliency_values = [row["saliency_concentration"] for row in group_rows if row.get("saliency_concentration") is not None]
        peak_values = [row["peak_inside_gt_box"] for row in group_rows if row.get("peak_inside_gt_box") is not None]
        row = {
            "group_name": group_name,
            "count": len(group_rows),
            **_prefix_dict("energy_in_gt_box", _summary_stats(gt_values)),
            **_prefix_dict("energy_in_pred_box", _summary_stats(pred_values)),
            "peak_inside_gt_box_rate": _mean_bool(peak_values),
            "peak_inside_gt_box_true_count": sum(1 for value in peak_values if value is True),
            "peak_inside_gt_box_observed_count": len(peak_values),
            **_prefix_dict("saliency_concentration", _summary_stats(saliency_values)),
        }
        summary_rows.append(row)
    return summary_rows


def select_representative_cases(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Pick high-signal rows for Phase 6 manual review."""

    selectors: list[tuple[str, str, list[dict[str, Any]], Callable[[dict[str, Any]], float]]] = [
        (
            "tp_high_evidence",
            "True-positive proxy rows with the highest GT-box evidence.",
            [row for row in rows if row.get("case_group") == "true_positive_proxy"],
            lambda row: _coalesce(row.get("energy_in_gt_box"), -1.0) * 3.0
            + _coalesce(row.get("energy_in_pred_box"), -1.0) * 2.0
            + _coalesce(row.get("saliency_concentration"), -1.0),
        ),
        (
            "fn_low_evidence",
            "False negatives where evidence inside the GT box is especially weak.",
            [row for row in rows if row.get("case_group") == "false_negative"],
            lambda row: -(
                _coalesce(row.get("energy_in_gt_box"), 1.0) * 3.0
                + _coalesce(row.get("saliency_concentration"), 1.0)
            ),
        ),
        (
            "fn_misaligned_evidence",
            "False negatives where evidence favors a predicted region or misses the GT box.",
            [row for row in rows if row.get("case_group") == "false_negative"],
            lambda row: (
                _coalesce(row.get("energy_in_pred_box"), 0.0) * 3.0
                - _coalesce(row.get("energy_in_gt_box"), 0.0) * 2.0
                + (0.2 if row.get("peak_inside_gt_box") is False else 0.0)
            ),
        ),
        (
            "fp_background_like_evidence",
            "False positives whose saliency is weak inside the predicted region.",
            [row for row in rows if row.get("case_group") == "false_positive"],
            lambda row: -(
                _coalesce(row.get("energy_in_pred_box"), 1.0) * 3.0
                + _coalesce(row.get("energy_in_gt_box"), 1.0)
            ) + _coalesce(row.get("saliency_concentration"), 0.0),
        ),
        (
            "localization_misaligned_evidence",
            "Localization errors where GT and predicted evidence appear separated.",
            [row for row in rows if row.get("case_group") == "localization_error"],
            lambda row: abs(_coalesce(row.get("energy_in_pred_box"), 0.0) - _coalesce(row.get("energy_in_gt_box"), 0.0)) * 3.0
            + (0.25 if row.get("peak_inside_gt_box") is False else 0.0),
        ),
        (
            "near_threshold_high_evidence",
            "Near-threshold rows with comparatively strong GT evidence.",
            [row for row in rows if row.get("is_near_threshold") is True],
            lambda row: _coalesce(row.get("energy_in_gt_box"), -1.0) * 3.0
            + _coalesce(row.get("saliency_concentration"), -1.0),
        ),
        (
            "near_threshold_low_evidence",
            "Near-threshold rows with comparatively weak GT evidence.",
            [row for row in rows if row.get("is_near_threshold") is True],
            lambda row: -(
                _coalesce(row.get("energy_in_gt_box"), 1.0) * 3.0
                + _coalesce(row.get("saliency_concentration"), 1.0)
            ),
        ),
    ]

    outputs: list[dict[str, Any]] = []
    for representative_group, rationale, candidate_rows, score_fn in selectors:
        ranked = sorted(candidate_rows, key=score_fn, reverse=True)
        for rank, row in enumerate(ranked[:top_k], start=1):
            outputs.append(
                {
                    "representative_group": representative_group,
                    "rank": rank,
                    "selection_score": round(float(score_fn(row)), 6),
                    "selection_rationale": rationale,
                    "case_id": row.get("case_id"),
                    "method": row.get("method"),
                    "focus_class": row.get("focus_class"),
                    "case_group": row.get("case_group"),
                    "is_near_threshold": row.get("is_near_threshold"),
                    "best_iou": row.get("best_iou"),
                    "confidence": row.get("confidence"),
                    "energy_in_gt_box": row.get("energy_in_gt_box"),
                    "energy_in_pred_box": row.get("energy_in_pred_box"),
                    "peak_inside_gt_box": row.get("peak_inside_gt_box"),
                    "saliency_concentration": row.get("saliency_concentration"),
                    "image_path": row.get("image_path"),
                    "overlay_path": row.get("overlay_path"),
                    "crop_path": row.get("crop_path"),
                    "map_path": row.get("map_path"),
                    "notes": row.get("notes"),
                }
            )
    return outputs


def _load_rows(evidence_csv: Path, focus_class: str | None, project_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    with evidence_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Evidence CSV has no header row: {evidence_csv}")
        fieldnames = set(reader.fieldnames)
        missing_optional = [column for column in OPTIONAL_METRIC_COLUMNS if column not in fieldnames]
        for column in missing_optional:
            message = f"Missing metric column '{column}' in evidence CSV; downstream fields will be null."
            warnings.append(message)
            warnings.warn(message)
        if "case_group" not in fieldnames:
            raise ValueError("Evidence CSV is missing required column 'case_group'.")
        if "focus_class" not in fieldnames and focus_class:
            message = "Missing 'focus_class' column; --focus-class filter was ignored."
            warnings.append(message)
            warnings.warn(message)
        raw_rows = list(reader)

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if focus_class and "focus_class" in raw_row and raw_row.get("focus_class") != focus_class:
            continue
        parsed = dict(raw_row)
        parsed["best_iou"] = _parse_float(raw_row.get("best_iou"))
        parsed["confidence"] = _parse_float(raw_row.get("confidence"))
        parsed["energy_in_gt_box"] = _parse_float(raw_row.get("energy_in_gt_box"))
        parsed["energy_in_pred_box"] = _parse_float(raw_row.get("energy_in_pred_box"))
        parsed["saliency_concentration"] = _parse_float(raw_row.get("saliency_concentration"))
        parsed["peak_value"] = _parse_float(raw_row.get("peak_value"))
        parsed["pixel_area"] = _parse_float(raw_row.get("pixel_area"))
        parsed["peak_x"] = _parse_int(raw_row.get("peak_x"))
        parsed["peak_y"] = _parse_int(raw_row.get("peak_y"))
        parsed["target_layer"] = _parse_int(raw_row.get("target_layer"))
        parsed["is_near_threshold"] = _parse_bool(raw_row.get("is_near_threshold"))
        parsed["peak_inside_gt_box"] = _parse_bool(raw_row.get("peak_inside_gt_box"))
        for path_field in ("image_path", "overlay_path", "crop_path", "map_path"):
            parsed[path_field] = _normalize_path(raw_row.get(path_field), project_root)
        rows.append(parsed)
    return rows, warnings


def _build_review_notes_template(representative_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template_rows: list[dict[str, Any]] = []
    for row in representative_rows:
        template_rows.append(
            {
                "representative_group": row["representative_group"],
                "rank": row["rank"],
                "case_id": row["case_id"],
                "case_group": row["case_group"],
                "overlay_path": row["overlay_path"],
                "crop_path": row["crop_path"],
                "map_path": row["map_path"],
                "review_status": "pending",
                "manual_pattern_label": "",
                "manual_observation": "",
                "phase7_relevance": "",
                "follow_up_note": "",
            }
        )
    return template_rows


def _maybe_write_plots(output_dir: Path, group_summary_rows: list[dict[str, Any]], warnings: list[str]) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        warnings.append("matplotlib is unavailable; optional Phase 6 plots were skipped.")
        return []

    plotted_rows = [row for row in group_summary_rows if row["count"] > 0]
    if not plotted_rows:
        return []

    outputs: list[Path] = []
    groups = [row["group_name"] for row in plotted_rows]
    gt_means = [row["energy_in_gt_box_mean"] or 0.0 for row in plotted_rows]
    pred_means = [row["energy_in_pred_box_mean"] or 0.0 for row in plotted_rows]
    saliency_means = [row["saliency_concentration_mean"] or 0.0 for row in plotted_rows]
    peak_rates = [row["peak_inside_gt_box_rate"] or 0.0 for row in plotted_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = range(len(groups))
    width = 0.25
    ax.bar([x - width for x in x_positions], gt_means, width=width, label="GT energy mean")
    ax.bar(x_positions, pred_means, width=width, label="Pred energy mean")
    ax.bar([x + width for x in x_positions], saliency_means, width=width, label="Saliency concentration mean")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Phase 6 evidence means by group")
    ax.legend()
    fig.tight_layout()
    mean_plot = output_dir / "evidence_group_means.png"
    fig.savefig(mean_plot, dpi=160)
    plt.close(fig)
    outputs.append(mean_plot)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(groups, peak_rates, color="#4C78A8")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Peak-inside-GT rate by group")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    peak_plot = output_dir / "peak_inside_gt_rate.png"
    fig.savefig(peak_plot, dpi=160)
    plt.close(fig)
    outputs.append(peak_plot)
    return outputs


def _write_readme(
    output_dir: Path,
    focus_class: str,
    top_k: int,
    warnings: list[str],
    group_summary_rows: list[dict[str, Any]],
    representative_rows: list[dict[str, Any]],
) -> Path:
    lines = [
        "# Phase 6 Evidence Review",
        "",
        "This directory contains descriptive post-hoc review artifacts derived from the Phase 5 evidence CSV.",
        "",
        "## Run settings",
        "",
        f"- Focus class: `{focus_class}`",
        f"- Representative cases per review bucket: `{top_k}`",
        f"- Review groups: `{', '.join(REQUIRED_GROUPS)}`",
        "",
        "## Output files",
        "",
        "- `evidence_group_summary.csv`",
        "- `evidence_group_summary.json`",
        "- `representative_cases.csv`",
        "- `review_notes_template.csv`",
        "- optional `*.png` plots when `matplotlib` is available",
        "",
        "## Group counts",
        "",
    ]
    for row in group_summary_rows:
        lines.append(f"- `{row['group_name']}`: `{row['count']}` rows")
    lines.extend(
        [
            "",
            "## Representative review buckets",
            "",
        ]
    )
    for group_name, count in _count_by_key(representative_rows, "representative_group").items():
        lines.append(f"- `{group_name}`: `{count}` rows")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- These outputs describe post-hoc evidence patterns only.",
            "- They do not establish that highlighted regions caused the detector prediction.",
            "- They should be used to guide manual review and Phase 7 scoping, not to claim causal mechanisms.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    output_path = output_dir / "README.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def _summary_stats(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    if not items:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    if len(items) == 1:
        std_value = 0.0
    else:
        std_value = statistics.stdev(items)
    return {
        "mean": round(float(statistics.mean(items)), 6),
        "median": round(float(statistics.median(items)), 6),
        "std": round(float(std_value), 6),
        "min": round(float(min(items)), 6),
        "max": round(float(max(items)), 6),
    }


def _prefix_dict(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _mean_bool(values: Iterable[bool]) -> float | None:
    items = list(values)
    if not items:
        return None
    return round(sum(1 for item in items if item is True) / len(items), 6)


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _normalize_path(value: str | None, project_root: Path) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    return _to_repo_relative(path, project_root)


def _to_repo_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _coalesce(value: float | None, fallback: float) -> float:
    return fallback if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts
