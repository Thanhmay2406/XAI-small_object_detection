"""Phase 8 intervention decision design from existing evidence artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


INTERVENTION_FAMILIES = [
    "DATA_AUDIT_OR_RELABEL",
    "DATA_SAMPLING_OR_CURRICULUM",
    "AUGMENTATION_REBALANCE",
    "LOSS_REWEIGHTING_PROTOTYPE",
    "SALIENCY_GUIDED_ATTENTION_PROTOTYPE",
    "NO_INTERVENTION_YET",
]


def design_phase8_interventions(
    representatives_csv: str | Path,
    group_summary_csv: str | Path,
    manual_review_summary_csv: str | Path,
    manual_review_filled_csv: str | Path,
    output_dir: str | Path,
    focus_class: str,
) -> dict[str, Path]:
    """Create Phase 8 decision-design artifacts without training or XAI changes."""

    warnings: list[str] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    representatives_path = Path(representatives_csv)
    group_summary_path = Path(group_summary_csv)
    manual_review_summary_path = Path(manual_review_summary_csv)
    manual_review_filled_path = Path(manual_review_filled_csv)
    manual_review_summary_json_path = manual_review_summary_path.with_suffix(".json")

    representatives = _read_optional_csv(representatives_path, warnings, "representatives")
    group_summary_rows = _read_optional_csv(group_summary_path, warnings, "group-summary")
    manual_review_filled_rows = _read_optional_csv(manual_review_filled_path, warnings, "manual-review-filled")
    _ = _read_optional_csv(manual_review_summary_path, warnings, "manual-review-summary")
    manual_review_summary_json = _read_optional_json(
        manual_review_summary_json_path,
        warnings,
        "manual-review-summary-json",
    )

    manual_review_meta = _assess_manual_review_source(
        manual_review_filled_path=manual_review_filled_path,
        manual_review_filled_rows=manual_review_filled_rows,
        manual_review_summary_json=manual_review_summary_json,
    )
    if manual_review_meta["manual_review_used_as_research_evidence"] is False:
        warnings.append(
            "Manual review is not verified as real research annotation; it is excluded from research-evidence decisions."
        )

    manual_review_by_group = _summarize_manual_review_by_group(manual_review_filled_rows)
    profiles = _build_evidence_profiles(representatives, group_summary_rows, warnings)
    decision_rows = _build_decision_rows(
        profiles=profiles,
        manual_review_meta=manual_review_meta,
        manual_review_by_group=manual_review_by_group,
        focus_class=focus_class,
        warnings=warnings,
    )
    candidate_rows = _build_candidate_rows(
        decision_rows=decision_rows,
        focus_class=focus_class,
        manual_review_meta=manual_review_meta,
    )
    no_intervention_rows = [
        row
        for row in decision_rows
        if row["intervention_family"] == "NO_INTERVENTION_YET"
        or row["evidence_strength"] in {"insufficient_evidence", "do_not_intervene_yet"}
    ]

    outputs = {
        "intervention_decision_table_csv": _write_csv(
            decision_rows,
            output_path / "intervention_decision_table.csv",
        ),
        "intervention_decision_table_json": _write_json(
            {
                "focus_class": focus_class,
                **manual_review_meta,
                "warnings": warnings,
                "decision_count": len(decision_rows),
                "decisions": decision_rows,
            },
            output_path / "intervention_decision_table.json",
        ),
        "intervention_candidates_csv": _write_csv(
            candidate_rows,
            output_path / "intervention_candidates.csv",
        ),
        "intervention_candidates_json": _write_json(
            {
                "focus_class": focus_class,
                **manual_review_meta,
                "warnings": warnings,
                "candidate_count": len(candidate_rows),
                "candidates": candidate_rows,
            },
            output_path / "intervention_candidates.json",
        ),
        "no_intervention_or_insufficient_evidence_csv": _write_csv(
            no_intervention_rows,
            output_path / "no_intervention_or_insufficient_evidence.csv",
        ),
        "readme_md": _write_readme(
            output_dir=output_path,
            focus_class=focus_class,
            decision_rows=decision_rows,
            candidate_rows=candidate_rows,
            manual_review_meta=manual_review_meta,
            warnings=warnings,
        ),
    }
    return outputs


def _build_evidence_profiles(
    representatives: list[dict[str, str]],
    group_summary_rows: list[dict[str, str]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if representatives:
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in representatives:
            evidence_group = row.get("representative_group") or row.get("case_group") or "unknown_group"
            by_group[evidence_group].append(row)

        profiles: list[dict[str, Any]] = []
        group_summary_index = {row.get("group_name", ""): row for row in group_summary_rows if row.get("group_name")}
        for evidence_group, rows in sorted(by_group.items()):
            case_group_counts = Counter(row.get("case_group", "") for row in rows if row.get("case_group"))
            dominant_case_group = case_group_counts.most_common(1)[0][0] if case_group_counts else ""
            summary_row = group_summary_index.get(dominant_case_group, {})
            gt_values = _collect_float_values(rows, "energy_in_gt_box")
            pred_values = _collect_float_values(rows, "energy_in_pred_box")
            saliency_values = _collect_float_values(rows, "saliency_concentration")
            peak_values = _collect_bool_values(rows, "peak_inside_gt_box")
            near_threshold_values = _collect_bool_values(rows, "is_near_threshold")
            profiles.append(
                {
                    "evidence_group": evidence_group,
                    "case_count": len(rows),
                    "case_group_counts": dict(case_group_counts),
                    "dominant_case_group": dominant_case_group or "unknown_case_group",
                    "mean_energy_in_gt_box": _safe_mean(gt_values),
                    "median_energy_in_gt_box": _safe_median(gt_values),
                    "mean_energy_in_pred_box": _safe_mean(pred_values),
                    "mean_saliency_concentration": _safe_mean(saliency_values),
                    "peak_inside_gt_box_rate": _safe_rate(peak_values),
                    "near_threshold_rate": _safe_rate(near_threshold_values),
                    "summary_count": _to_int(summary_row.get("count")),
                    "summary_energy_in_gt_box_mean": _to_float(summary_row.get("energy_in_gt_box_mean")),
                    "summary_energy_in_pred_box_mean": _to_float(summary_row.get("energy_in_pred_box_mean")),
                    "summary_peak_inside_gt_box_rate": _to_float(summary_row.get("peak_inside_gt_box_rate")),
                    "summary_saliency_concentration_mean": _to_float(summary_row.get("saliency_concentration_mean")),
                    "missing_metrics": _missing_metric_names(
                        gt_values=gt_values,
                        pred_values=pred_values,
                        saliency_values=saliency_values,
                        peak_values=peak_values,
                    ),
                }
            )
        return profiles

    if group_summary_rows:
        warnings.append("Representative cases are missing; Phase 8 decisions will use group-summary rows only.")
        profiles = []
        for row in group_summary_rows:
            profiles.append(
                {
                    "evidence_group": row.get("group_name", "unknown_group"),
                    "case_count": _to_int(row.get("count")) or 0,
                    "case_group_counts": {row.get("group_name", "unknown_group"): _to_int(row.get("count")) or 0},
                    "dominant_case_group": row.get("group_name", "unknown_case_group"),
                    "mean_energy_in_gt_box": _to_float(row.get("energy_in_gt_box_mean")),
                    "median_energy_in_gt_box": _to_float(row.get("energy_in_gt_box_median")),
                    "mean_energy_in_pred_box": _to_float(row.get("energy_in_pred_box_mean")),
                    "mean_saliency_concentration": _to_float(row.get("saliency_concentration_mean")),
                    "peak_inside_gt_box_rate": _to_float(row.get("peak_inside_gt_box_rate")),
                    "near_threshold_rate": None,
                    "summary_count": _to_int(row.get("count")),
                    "summary_energy_in_gt_box_mean": _to_float(row.get("energy_in_gt_box_mean")),
                    "summary_energy_in_pred_box_mean": _to_float(row.get("energy_in_pred_box_mean")),
                    "summary_peak_inside_gt_box_rate": _to_float(row.get("peak_inside_gt_box_rate")),
                    "summary_saliency_concentration_mean": _to_float(row.get("saliency_concentration_mean")),
                    "missing_metrics": _missing_metric_names(
                        gt_values=[] if _to_float(row.get("energy_in_gt_box_mean")) is None else [0.0],
                        pred_values=[] if _to_float(row.get("energy_in_pred_box_mean")) is None else [0.0],
                        saliency_values=[] if _to_float(row.get("saliency_concentration_mean")) is None else [0.0],
                        peak_values=[] if _to_float(row.get("peak_inside_gt_box_rate")) is None else [True],
                    ),
                }
            )
        return profiles

    warnings.append("No representative or group-summary evidence inputs were found; placeholder outputs will be written.")
    return [
        {
            "evidence_group": "no_evidence_inputs",
            "case_count": 0,
            "case_group_counts": {},
            "dominant_case_group": "unknown_case_group",
            "mean_energy_in_gt_box": None,
            "median_energy_in_gt_box": None,
            "mean_energy_in_pred_box": None,
            "mean_saliency_concentration": None,
            "peak_inside_gt_box_rate": None,
            "near_threshold_rate": None,
            "summary_count": None,
            "summary_energy_in_gt_box_mean": None,
            "summary_energy_in_pred_box_mean": None,
            "summary_peak_inside_gt_box_rate": None,
            "summary_saliency_concentration_mean": None,
            "missing_metrics": [
                "energy_in_gt_box",
                "energy_in_pred_box",
                "saliency_concentration",
                "peak_inside_gt_box",
            ],
        }
    ]


def _build_decision_rows(
    profiles: list[dict[str, Any]],
    manual_review_meta: dict[str, Any],
    manual_review_by_group: dict[str, dict[str, Any]],
    focus_class: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, profile in enumerate(profiles, start=1):
        manual_group = manual_review_by_group.get(profile["evidence_group"], {})
        evidence_strength = _classify_evidence_strength(profile, manual_review_meta, manual_group)
        family, recommended_action, safe_to_prototype, priority = _decide_intervention(
            profile=profile,
            evidence_strength=evidence_strength,
            manual_review_meta=manual_review_meta,
            manual_group=manual_group,
        )
        if profile["missing_metrics"]:
            warnings.append(
                f"{profile['evidence_group']}: missing metrics {', '.join(profile['missing_metrics'])}; decision quality is degraded."
            )
        rows.append(
            {
                "decision_id": f"phase8_decision_{idx:03d}",
                "focus_class": focus_class,
                "evidence_group": profile["evidence_group"],
                "case_count": profile["case_count"],
                "evidence_strength": evidence_strength,
                "manual_review_status": _manual_review_status(manual_review_meta, manual_group),
                "recommended_action": recommended_action,
                "intervention_family": family,
                "safe_to_prototype": safe_to_prototype,
                "priority": priority,
                "rationale": _build_rationale(profile, manual_review_meta, manual_group, evidence_strength),
                "required_next_evidence": _required_next_evidence(profile, manual_review_meta, family),
                "notes": _build_notes(profile, manual_review_meta, manual_group),
                "dominant_case_group": profile["dominant_case_group"],
                "mean_energy_in_gt_box": _format_optional_float(profile["mean_energy_in_gt_box"]),
                "mean_energy_in_pred_box": _format_optional_float(profile["mean_energy_in_pred_box"]),
                "mean_saliency_concentration": _format_optional_float(profile["mean_saliency_concentration"]),
                "peak_inside_gt_box_rate": _format_optional_float(profile["peak_inside_gt_box_rate"]),
                "near_threshold_rate": _format_optional_float(profile["near_threshold_rate"]),
                "manual_review_row_count": manual_group.get("row_count", 0),
                "manual_review_top_label_quality": manual_group.get("top_label_quality", ""),
                "manual_review_top_recommended_action": manual_group.get("top_recommended_action", ""),
            }
        )
    return rows


def _build_candidate_rows(
    decision_rows: list[dict[str, Any]],
    focus_class: str,
    manual_review_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        family = row["intervention_family"]
        if family == "NO_INTERVENTION_YET":
            continue
        grouped[family].append(row)

    outputs: list[dict[str, Any]] = []
    for idx, (family, rows) in enumerate(sorted(grouped.items()), start=1):
        target_failure_mode = ", ".join(sorted({row["dominant_case_group"] for row in rows if row["dominant_case_group"]}))
        evidence_groups = ", ".join(sorted({row["evidence_group"] for row in rows}))
        outputs.append(
            {
                "candidate_id": f"phase9_candidate_{idx:03d}",
                "focus_class": focus_class,
                "intervention_family": family,
                "target_failure_mode": target_failure_mode,
                "hypothesis": _candidate_hypothesis(family, target_failure_mode),
                "expected_effect": _candidate_expected_effect(family),
                "required_inputs": _candidate_required_inputs(evidence_groups),
                "implementation_risk": _candidate_risk(family),
                "evaluation_needed": _candidate_evaluation_needed(family),
                "should_implement_now": "prototype_only_after_real_review",
                "reason": _candidate_reason(rows, manual_review_meta),
                "evidence_groups": evidence_groups,
            }
        )
    return outputs


def _assess_manual_review_source(
    manual_review_filled_path: Path,
    manual_review_filled_rows: list[dict[str, str]],
    manual_review_summary_json: dict[str, Any] | None,
) -> dict[str, Any]:
    if not manual_review_filled_rows:
        return {
            "manual_review_source": "missing",
            "manual_review_used_as_research_evidence": False,
            "manual_review_reason": "No manual_review_filled.csv rows were available.",
        }

    review_text = manual_review_filled_path.read_text(encoding="utf-8") if manual_review_filled_path.exists() else ""
    lowered = review_text.lower()
    summary_notes = " ".join(str(note) for note in (manual_review_summary_json or {}).get("notes", [])).lower()
    if any(token in lowered for token in ["synthetic", "smoke-test", "demo annotation", "demo/synthetic"]):
        return {
            "manual_review_source": "demo_or_synthetic",
            "manual_review_used_as_research_evidence": False,
            "manual_review_reason": "manual_review_filled.csv contains explicit synthetic/demo smoke-test markers.",
        }
    if any(token in summary_notes for token in ["not causal proof", "support phase 8 intervention design review only"]):
        return {
            "manual_review_source": "unknown",
            "manual_review_used_as_research_evidence": False,
            "manual_review_reason": "Manual review exists but is not verified as a real research annotation set.",
        }
    return {
        "manual_review_source": "unknown",
        "manual_review_used_as_research_evidence": False,
        "manual_review_reason": "Manual review content was found, but Phase 8 uses a conservative non-research-evidence setting.",
    }


def _summarize_manual_review_by_group(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = row.get("bucket") or row.get("representative_group") or "unknown_group"
        by_group[group].append(row)

    outputs: dict[str, dict[str, Any]] = {}
    for group, group_rows in by_group.items():
        label_counter = Counter(row.get("label_quality", "") for row in group_rows if row.get("label_quality"))
        action_counter = Counter(row.get("recommended_action", "") for row in group_rows if row.get("recommended_action"))
        alignment_counter = Counter(row.get("saliency_alignment", "") for row in group_rows if row.get("saliency_alignment"))
        outputs[group] = {
            "row_count": len(group_rows),
            "top_label_quality": label_counter.most_common(1)[0][0] if label_counter else "",
            "top_recommended_action": action_counter.most_common(1)[0][0] if action_counter else "",
            "top_saliency_alignment": alignment_counter.most_common(1)[0][0] if alignment_counter else "",
            "questionable_label_share": _counter_share(label_counter, {"questionable", "likely_noise"}),
            "uncertain_alignment_share": _counter_share(alignment_counter, {"uncertain", "misaligned", "absent"}),
        }
    return outputs


def _classify_evidence_strength(
    profile: dict[str, Any],
    manual_review_meta: dict[str, Any],
    manual_group: dict[str, Any],
) -> str:
    metrics_present = 0
    score = 0.0

    case_count = profile.get("case_count", 0) or 0
    if case_count >= 6:
        score += 1.0
        metrics_present += 1
    elif case_count >= 3:
        score += 0.5
        metrics_present += 1

    mean_gt = _first_not_none(profile.get("mean_energy_in_gt_box"), profile.get("summary_energy_in_gt_box_mean"))
    if mean_gt is not None:
        metrics_present += 1
        if mean_gt >= 0.15:
            score += 1.0
        elif mean_gt >= 0.03:
            score += 0.5

    saliency_mean = _first_not_none(
        profile.get("mean_saliency_concentration"),
        profile.get("summary_saliency_concentration_mean"),
    )
    if saliency_mean is not None:
        metrics_present += 1
        if saliency_mean >= 0.40:
            score += 1.0
        elif saliency_mean >= 0.25:
            score += 0.5

    peak_rate = _first_not_none(profile.get("peak_inside_gt_box_rate"), profile.get("summary_peak_inside_gt_box_rate"))
    if peak_rate is not None:
        metrics_present += 1
        if peak_rate >= 0.50:
            score += 1.0
        elif peak_rate >= 0.20:
            score += 0.5

    if profile.get("near_threshold_rate") and profile["near_threshold_rate"] >= 0.5:
        score -= 0.25

    dominant_case_group = str(profile.get("dominant_case_group", ""))
    evidence_group = str(profile.get("evidence_group", ""))
    if "misaligned" in evidence_group or dominant_case_group == "localization_error":
        score -= 0.25

    if metrics_present < 2 or case_count < 3:
        return "insufficient_evidence"
    if dominant_case_group == "false_positive" and (mean_gt or 0.0) >= 0.15:
        return "do_not_intervene_yet"
    if manual_group.get("questionable_label_share", 0.0) >= 0.3:
        return "do_not_intervene_yet"
    if manual_review_meta["manual_review_used_as_research_evidence"] is False:
        return "weak_or_mixed_xai_support" if score >= 2.0 else "insufficient_evidence"
    if score >= 3.0 and manual_group.get("uncertain_alignment_share", 0.0) < 0.25:
        return "strong_xai_support"
    if score >= 1.5:
        return "weak_or_mixed_xai_support"
    return "insufficient_evidence"


def _decide_intervention(
    profile: dict[str, Any],
    evidence_strength: str,
    manual_review_meta: dict[str, Any],
    manual_group: dict[str, Any],
) -> tuple[str, str, str, str]:
    if evidence_strength in {"insufficient_evidence", "do_not_intervene_yet"}:
        return ("NO_INTERVENTION_YET", "collect_real_manual_review_first", "false", "low")

    case_group = str(profile.get("dominant_case_group", ""))
    evidence_group = str(profile.get("evidence_group", ""))
    gate = (
        "prototype_only_after_real_review"
        if manual_review_meta["manual_review_used_as_research_evidence"] is False
        else "true"
    )

    if manual_group.get("questionable_label_share", 0.0) > 0.0 or "label" in evidence_group:
        return ("DATA_AUDIT_OR_RELABEL", "prepare_data_audit_plan", gate, "high")
    if case_group == "false_positive" or "background" in evidence_group:
        return ("AUGMENTATION_REBALANCE", "prepare_augmentation_rebalance_hypothesis", gate, "medium")
    if "near_threshold" in evidence_group or case_group == "true_positive_proxy":
        return ("DATA_SAMPLING_OR_CURRICULUM", "prepare_sampling_curriculum_hypothesis", gate, "medium")
    if case_group in {"false_negative", "localization_error"}:
        mean_gt = _first_not_none(profile.get("mean_energy_in_gt_box"), profile.get("summary_energy_in_gt_box_mean")) or 0.0
        if mean_gt <= 0.03:
            return ("LOSS_REWEIGHTING_PROTOTYPE", "prepare_loss_reweighting_prototype", gate, "medium")
        return (
            "SALIENCY_GUIDED_ATTENTION_PROTOTYPE",
            "prepare_saliency_guided_attention_prototype",
            gate,
            "medium",
        )
    return ("NO_INTERVENTION_YET", "defer_intervention", "false", "low")


def _build_rationale(
    profile: dict[str, Any],
    manual_review_meta: dict[str, Any],
    manual_group: dict[str, Any],
    evidence_strength: str,
) -> str:
    parts = [f"group={profile['evidence_group']}", f"case_count={profile['case_count']}"]
    for key in [
        "mean_energy_in_gt_box",
        "mean_energy_in_pred_box",
        "mean_saliency_concentration",
        "peak_inside_gt_box_rate",
    ]:
        value = profile.get(key)
        if value is not None:
            parts.append(f"{key}={value:.4f}")
    if profile.get("missing_metrics"):
        parts.append(f"missing_metrics={','.join(profile['missing_metrics'])}")
    parts.append(f"evidence_strength={evidence_strength}")
    parts.append(f"manual_review_source={manual_review_meta['manual_review_source']}")
    if manual_group:
        parts.append(f"manual_review_row_count={manual_group.get('row_count', 0)}")
        if manual_group.get("top_saliency_alignment"):
            parts.append(f"top_saliency_alignment={manual_group['top_saliency_alignment']}")
    return "; ".join(parts)


def _required_next_evidence(
    profile: dict[str, Any],
    manual_review_meta: dict[str, Any],
    family: str,
) -> str:
    needs = []
    if manual_review_meta["manual_review_used_as_research_evidence"] is False:
        needs.append("real manual review on the same representative buckets")
    if profile.get("missing_metrics"):
        needs.append(f"fill missing metrics: {', '.join(profile['missing_metrics'])}")
    if family == "DATA_AUDIT_OR_RELABEL":
        needs.append("targeted label audit or ambiguity review")
    elif family == "DATA_SAMPLING_OR_CURRICULUM":
        needs.append("bucket-level size and difficulty distribution check")
    elif family == "AUGMENTATION_REBALANCE":
        needs.append("failure cases grouped by scale, blur, and background clutter")
    elif family == "LOSS_REWEIGHTING_PROTOTYPE":
        needs.append("repeatable evidence pattern across real false-negative review rows")
    elif family == "SALIENCY_GUIDED_ATTENTION_PROTOTYPE":
        needs.append("cross-method XAI confirmation before any prototype")
    if not needs:
        needs.append("more stable evidence before Phase 9")
    return "; ".join(needs)


def _build_notes(
    profile: dict[str, Any],
    manual_review_meta: dict[str, Any],
    manual_group: dict[str, Any],
) -> str:
    notes = []
    if manual_review_meta["manual_review_used_as_research_evidence"] is False:
        notes.append("manual review not used as research evidence")
    if profile.get("dominant_case_group") == "false_positive":
        notes.append("false-positive evidence may still reflect label or overlap ambiguity")
    if profile.get("near_threshold_rate") and profile["near_threshold_rate"] >= 0.5:
        notes.append("near-threshold rows are inherently borderline and should stay descriptive")
    if manual_group.get("questionable_label_share", 0.0) > 0.0:
        notes.append("manual-review label-quality flags suggest audit before intervention design")
    return "; ".join(notes)


def _manual_review_status(manual_review_meta: dict[str, Any], manual_group: dict[str, Any]) -> str:
    source = manual_review_meta["manual_review_source"]
    if source == "missing":
        return "missing"
    if source == "demo_or_synthetic":
        return f"demo_or_synthetic_not_used_rows={manual_group.get('row_count', 0)}"
    if source == "unknown":
        return f"unknown_not_used_rows={manual_group.get('row_count', 0)}"
    return f"verified_real_review_rows={manual_group.get('row_count', 0)}"


def _candidate_hypothesis(family: str, target_failure_mode: str) -> str:
    mapping = {
        "DATA_AUDIT_OR_RELABEL": f"If {target_failure_mode} is driven by label ambiguity, auditing or relabeling may reduce noisy supervision.",
        "DATA_SAMPLING_OR_CURRICULUM": f"If {target_failure_mode} clusters around difficult or near-threshold cases, sampling or curriculum changes may improve exposure.",
        "AUGMENTATION_REBALANCE": f"If {target_failure_mode} reflects scale, blur, crop, or clutter mismatch, augmentation rebalance may better cover those conditions.",
        "LOSS_REWEIGHTING_PROTOTYPE": f"If {target_failure_mode} repeatedly shows weak object evidence, selective reweighting might focus optimization on those misses.",
        "SALIENCY_GUIDED_ATTENTION_PROTOTYPE": f"If {target_failure_mode} shows consistent XAI misalignment across real review, a saliency-guided prototype may be worth studying later.",
    }
    return mapping.get(family, "No intervention hypothesis should be implemented yet.")


def _candidate_expected_effect(family: str) -> str:
    mapping = {
        "DATA_AUDIT_OR_RELABEL": "Cleaner supervision and fewer ambiguity-driven failure interpretations.",
        "DATA_SAMPLING_OR_CURRICULUM": "Better coverage of difficult small-object or borderline cases.",
        "AUGMENTATION_REBALANCE": "Improved robustness to background clutter, crop variation, blur, or scale shifts.",
        "LOSS_REWEIGHTING_PROTOTYPE": "Potentially stronger optimization focus on repeated miss patterns.",
        "SALIENCY_GUIDED_ATTENTION_PROTOTYPE": "Potentially tighter evidence concentration on relevant object regions.",
    }
    return mapping.get(family, "No expected effect should be claimed yet.")


def _candidate_required_inputs(evidence_groups: str) -> str:
    return (
        f"Phase 6 representative buckets: {evidence_groups}; real manual review; "
        "baseline evaluation artifacts; no training changes in Phase 8."
    )


def _candidate_risk(family: str) -> str:
    mapping = {
        "DATA_AUDIT_OR_RELABEL": "medium",
        "DATA_SAMPLING_OR_CURRICULUM": "medium",
        "AUGMENTATION_REBALANCE": "medium",
        "LOSS_REWEIGHTING_PROTOTYPE": "high",
        "SALIENCY_GUIDED_ATTENTION_PROTOTYPE": "high",
    }
    return mapping.get(family, "unknown")


def _candidate_evaluation_needed(family: str) -> str:
    if family in {"LOSS_REWEIGHTING_PROTOTYPE", "SALIENCY_GUIDED_ATTENTION_PROTOTYPE"}:
        return "real manual review, ablation plan, and baseline-vs-prototype comparison before any implementation claim"
    return "real manual review plus offline design review before moving to Phase 9"


def _candidate_reason(rows: list[dict[str, Any]], manual_review_meta: dict[str, Any]) -> str:
    evidence_groups = ", ".join(sorted(row["evidence_group"] for row in rows))
    return (
        f"Derived from evidence groups [{evidence_groups}] with manual_review_source={manual_review_meta['manual_review_source']}. "
        "Candidate remains hypothesis-level only."
    )


def _read_optional_csv(path: Path, warnings: list[str], label: str) -> list[dict[str, str]]:
    if not path.exists():
        warnings.append(f"Missing {label} input: {path}")
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_optional_json(path: Path, warnings: list[str], label: str) -> dict[str, Any] | None:
    if not path.exists():
        warnings.append(f"Missing {label} input: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    if not rows:
        rows = [{"status": "no_rows_generated"}]
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_readme(
    output_dir: Path,
    focus_class: str,
    decision_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    manual_review_meta: dict[str, Any],
    warnings: list[str],
) -> Path:
    family_counts = Counter(row["intervention_family"] for row in decision_rows if row.get("intervention_family"))
    strength_counts = Counter(row["evidence_strength"] for row in decision_rows if row.get("evidence_strength"))
    lines = [
        "# Phase 8 Intervention Decision Design",
        "",
        f"This directory contains Phase 8 decision-design artifacts for the `{focus_class}` evidence workflow.",
        "",
        "## Scope boundary",
        "",
        "- No training was run.",
        "- No model architecture was changed.",
        "- No training loop or loss implementation was changed.",
        "- No XAI extraction logic was changed.",
        "- These outputs are intervention-design hypotheses only for a later phase.",
        "",
        "## Manual review handling",
        "",
        f"- `manual_review_source = {manual_review_meta['manual_review_source']}`",
        f"- `manual_review_used_as_research_evidence = {str(manual_review_meta['manual_review_used_as_research_evidence']).lower()}`",
        f"- reason: {manual_review_meta['manual_review_reason']}",
        "",
        "## Output summary",
        "",
        f"- decisions: `{len(decision_rows)}`",
        f"- candidate interventions: `{len(candidate_rows)}`",
    ]
    for strength, count in sorted(strength_counts.items()):
        lines.append(f"- evidence strength `{strength}`: `{count}`")
    for family, count in sorted(family_counts.items()):
        lines.append(f"- intervention family `{family}`: `{count}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- If manual review is demo/synthetic or unknown, the outputs confirm only the code path and decision scaffold, not real research evidence.",
            "- Candidate interventions must not be described as validated improvements.",
            "- Phase 9 should begin only after real manual review and stronger cross-bucket evidence support are available.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    output_path = output_dir / "README.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _collect_float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _to_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _collect_bool_values(rows: list[dict[str, str]], key: str) -> list[bool]:
    values = []
    for row in rows:
        value = _to_bool(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _missing_metric_names(
    gt_values: list[float],
    pred_values: list[float],
    saliency_values: list[float],
    peak_values: list[bool],
) -> list[str]:
    missing = []
    if not gt_values:
        missing.append("energy_in_gt_box")
    if not pred_values:
        missing.append("energy_in_pred_box")
    if not saliency_values:
        missing.append("saliency_concentration")
    if not peak_values:
        missing.append("peak_inside_gt_box")
    return missing


def _counter_share(counter: Counter[str], target_values: set[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    matched = sum(count for key, count in counter.items() if key in target_values)
    return matched / total


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return median(values) if values else None


def _safe_rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"
