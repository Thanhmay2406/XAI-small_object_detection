"""Prepare a non-executing M8_v1c automated XAI policy-mining package.

This script does not run training, evaluation, fresh inference, checkpoint
loading, or dataset mutation. It only reads existing M8 evidence artifacts and
converts them into a machine-readable policy-mining proposal package.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

DECISION_SUMMARY_PATH = Path(
    "artifacts/m8_1_size_aware_fpn_scale_selection_decision_gate/m8_1_decision_summary.json"
)
M8_V1B_SUMMARY_PATH = Path(
    "artifacts/m8_v1b_size_aware_scale_weighting_implementation/m8_v1b_implementation_summary.json"
)
VAL_SUMMARY_PATH = Path("artifacts/m8_xai_fpn_scale_selection_val_full/m8_scale_summary.json")
TEST_SUMMARY_PATH = Path("artifacts/m8_xai_fpn_scale_selection_test_full/m8_scale_summary.json")
OUTPUT_DIR = Path("artifacts/m8_v1c_automated_policy_mining_preparation")

SCALE_ORDER = ("P2", "P3", "P4", "P5")
SIZE_GROUPS = ("small", "medium", "large")
FALLBACK_GROUPS = ("tiny", "unknown")
MIN_OBJECTS_FOR_AUTO_POLICY = 25
MIN_MARGIN_FOR_AUTO_POLICY = 0.10


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def ensure_upstream_statuses(
    decision_summary: dict[str, Any],
    v1b_summary: dict[str, Any],
    val_summary: dict[str, Any],
    test_summary: dict[str, Any],
) -> None:
    if decision_summary.get("status") != "m8_1_decision_gate_prepared":
        raise ValueError(
            "M8.1 decision summary must have status 'm8_1_decision_gate_prepared', "
            f"got {decision_summary.get('status')!r}."
        )
    if v1b_summary.get("status") != "m8_v1b_size_aware_scale_weighting_implementation_prepared":
        raise ValueError(
            "M8_v1b implementation summary must have status "
            "'m8_v1b_size_aware_scale_weighting_implementation_prepared', "
            f"got {v1b_summary.get('status')!r}."
        )
    for label, payload in (("val", val_summary), ("test", test_summary)):
        if payload.get("status") != "m8_scale_diagnosis_completed":
            raise ValueError(
                f"M8 scale summary for {label} must have status 'm8_scale_diagnosis_completed', "
                f"got {payload.get('status')!r}."
            )


def get_group_payload(summary: dict[str, Any], group_name: str) -> dict[str, Any]:
    groups = summary.get("object_size_group_to_dominant_scale", {})
    if not isinstance(groups, dict):
        return {}
    payload = groups.get(group_name, {})
    return payload if isinstance(payload, dict) else {}


def normalize_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        return {scale: 0 for scale in SCALE_ORDER}
    return {scale: int(counts.get(scale, 0) or 0) for scale in SCALE_ORDER}


def ranked_scales(counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def build_weight_delta(preferred_scale: str) -> dict[str, float]:
    ranked = [preferred_scale] + [scale for scale in SCALE_ORDER if scale != preferred_scale]
    values = (1.20, 1.00, 0.95, 0.90)
    return {scale: value for scale, value in zip(ranked, values)}


def summarize_group(
    group_name: str,
    val_summary: dict[str, Any],
    test_summary: dict[str, Any],
) -> dict[str, Any]:
    val_payload = get_group_payload(val_summary, group_name)
    test_payload = get_group_payload(test_summary, group_name)
    val_counts = normalize_counts(val_payload)
    test_counts = normalize_counts(test_payload)
    combined_counts = {scale: val_counts[scale] + test_counts[scale] for scale in SCALE_ORDER}

    val_total = int(val_payload.get("total_objects", 0) or 0)
    test_total = int(test_payload.get("total_objects", 0) or 0)
    combined_total = val_total + test_total

    combined_ranked = ranked_scales(combined_counts)
    best_scale, best_count = combined_ranked[0]
    second_scale, second_count = combined_ranked[1]
    margin = 0.0 if combined_total <= 0 else (best_count - second_count) / combined_total

    val_dominant = str(val_payload.get("dominant_scale", "unknown"))
    test_dominant = str(test_payload.get("dominant_scale", "unknown"))
    split_consistent = (
        val_dominant in SCALE_ORDER and test_dominant in SCALE_ORDER and val_dominant == test_dominant == best_scale
    )
    auto_policy_allowed = (
        combined_total >= MIN_OBJECTS_FOR_AUTO_POLICY
        and margin >= MIN_MARGIN_FOR_AUTO_POLICY
        and split_consistent
    )

    if auto_policy_allowed:
        mode = "preferred_scale"
        preferred_scale = best_scale
        rationale = (
            f"{group_name} is split-consistent and the best-vs-second margin is {margin:.3f}, "
            f"so {best_scale} is auto-proposed."
        )
        fallback_reason = ""
        weight_delta = build_weight_delta(best_scale)
    else:
        mode = "identity"
        preferred_scale = None
        rationale = (
            f"{group_name} stays identity because combined_total={combined_total}, "
            f"margin={margin:.3f}, split_consistent={split_consistent}."
        )
        fallback_reason = rationale
        weight_delta = {scale: 1.00 for scale in SCALE_ORDER}

    return {
        "size_group": group_name,
        "val_total_objects": val_total,
        "test_total_objects": test_total,
        "combined_total_objects": combined_total,
        "val_dominant_scale": val_dominant,
        "test_dominant_scale": test_dominant,
        "combined_counts": combined_counts,
        "combined_ranked_scales": [
            {"scale": scale, "count": count} for scale, count in combined_ranked
        ],
        "best_scale": best_scale,
        "second_best_scale": second_scale,
        "best_minus_second_margin": margin,
        "split_consistent": split_consistent,
        "auto_policy_allowed": auto_policy_allowed,
        "mode": mode,
        "preferred_scale": preferred_scale,
        "weight_delta": weight_delta,
        "rationale": rationale,
        "fallback_reason": fallback_reason,
    }


def build_artifact_rows(group_summaries: dict[str, dict[str, Any]]) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
    evidence_rows: list[list[Any]] = []
    fallback_rows: list[list[Any]] = []
    review_rows: list[list[Any]] = []

    for group_name in SIZE_GROUPS:
        summary = group_summaries[group_name]
        combined_total = summary["combined_total_objects"]
        for rank_index, ranked in enumerate(summary["combined_ranked_scales"], start=1):
            scale = ranked["scale"]
            combined_count = ranked["count"]
            share = 0.0 if combined_total <= 0 else combined_count / combined_total
            evidence_rows.append(
                [
                    group_name,
                    scale,
                    summary["val_dominant_scale"],
                    summary["test_dominant_scale"],
                    summary["val_total_objects"],
                    summary["test_total_objects"],
                    summary["combined_counts"][scale],
                    f"{share:.6f}",
                    rank_index,
                ]
            )

        review_rows.append(
            [
                group_name,
                summary["mode"],
                summary["preferred_scale"] or "identity",
                f"{summary['best_minus_second_margin']:.6f}",
                summary["combined_total_objects"],
                summary["split_consistent"],
                "human_review_required",
                "",
                "",
            ]
        )

        if summary["mode"] == "identity":
            fallback_rows.append(
                [
                    group_name,
                    summary["combined_total_objects"],
                    summary["val_dominant_scale"],
                    summary["test_dominant_scale"],
                    f"{summary['best_minus_second_margin']:.6f}",
                    summary["fallback_reason"],
                ]
            )

    fallback_rows.extend(
        [
            [
                "tiny",
                11,
                "P3",
                "P2",
                "0.000000",
                "Tiny group remains identity because sample size is too small and split dominance is inconsistent.",
            ],
            [
                "unknown",
                0,
                "unknown",
                "unknown",
                "0.000000",
                "Unknown size metadata must remain identity to fail closed.",
            ],
        ]
    )

    return evidence_rows, fallback_rows, review_rows


def main() -> None:
    decision_summary = load_json(DECISION_SUMMARY_PATH)
    v1b_summary = load_json(M8_V1B_SUMMARY_PATH)
    val_summary = load_json(VAL_SUMMARY_PATH)
    test_summary = load_json(TEST_SUMMARY_PATH)
    ensure_upstream_statuses(decision_summary, v1b_summary, val_summary, test_summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()

    group_summaries = {
        group_name: summarize_group(group_name, val_summary, test_summary) for group_name in SIZE_GROUPS
    }
    evidence_rows, fallback_rows, review_rows = build_artifact_rows(group_summaries)

    write_csv(
        OUTPUT_DIR / "scale_evidence_by_size.csv",
        [
            "size_group",
            "scale",
            "val_dominant_scale",
            "test_dominant_scale",
            "val_total_objects",
            "test_total_objects",
            "combined_count",
            "combined_share",
            "rank_within_size_group",
        ],
        evidence_rows,
    )

    write_csv(
        OUTPUT_DIR / "identity_fallback_cases.csv",
        [
            "size_group",
            "combined_total_objects",
            "val_dominant_scale",
            "test_dominant_scale",
            "best_minus_second_margin",
            "fallback_reason",
        ],
        fallback_rows,
    )

    write_csv(
        OUTPUT_DIR / "human_review_policy_template.csv",
        [
            "size_group",
            "auto_mode",
            "auto_preferred_scale",
            "best_minus_second_margin",
            "combined_total_objects",
            "split_consistent",
            "review_status",
            "reviewer_decision",
            "reviewer_notes",
        ],
        review_rows,
    )

    policy_candidate = {
        "method_name": "m8_v1c_automated_xai_policy_mining",
        "status": "proposal_from_existing_m8_evidence_only",
        "generated_at_utc": generated_at,
        "source_evidence_type": "dominant_scale_frequency_only",
        "policy_inference_rule": {
            "expression": "argmax_scale EvidenceScore(size_group, scale)",
            "minimum_objects_for_auto_policy": MIN_OBJECTS_FOR_AUTO_POLICY,
            "minimum_best_minus_second_margin": MIN_MARGIN_FOR_AUTO_POLICY,
            "requires_split_consistency": True,
            "identity_fallback_on_uncertainty": True,
        },
        "size_aware_policy": {},
        "fallback_groups": {
            "tiny": {
                "mode": "identity",
                "reason": "insufficient sample size and inconsistent split-level dominance",
            },
            "unknown": {
                "mode": "identity",
                "reason": "missing or invalid size metadata",
            },
        },
        "guardrails": {
            "training_execution_allowed": False,
            "evaluation_execution_allowed": False,
            "inference_execution_allowed": False,
            "dataset_mutation_allowed": False,
            "checkpoint_mutation_allowed": False,
        },
    }

    for group_name in SIZE_GROUPS:
        summary = group_summaries[group_name]
        if summary["mode"] == "preferred_scale":
            policy_candidate["size_aware_policy"][group_name] = {
                "mode": "preferred_scale",
                "preferred_scale": summary["preferred_scale"],
                "weight_delta": summary["weight_delta"],
                "confidence_margin": round(summary["best_minus_second_margin"], 6),
                "combined_total_objects": summary["combined_total_objects"],
                "rationale": summary["rationale"],
            }
        else:
            policy_candidate["size_aware_policy"][group_name] = {
                "mode": "identity",
                "weight_delta": summary["weight_delta"],
                "confidence_margin": round(summary["best_minus_second_margin"], 6),
                "combined_total_objects": summary["combined_total_objects"],
                "rationale": summary["rationale"],
            }
    policy_candidate["size_aware_policy"]["unknown"] = {
        "mode": "identity",
        "weight_delta": {scale: 1.00 for scale in SCALE_ORDER},
        "rationale": "Fail closed when size metadata is unavailable.",
    }
    write_yaml(OUTPUT_DIR / "policy_candidate.yaml", policy_candidate)

    confidence_report = {
        "phase": "M8_v1c",
        "status": "m8_v1c_policy_candidate_prepared",
        "generated_at_utc": generated_at,
        "upstream_dependencies_verified": {
            "m8_1_decision_summary": str(DECISION_SUMMARY_PATH),
            "m8_v1b_implementation_summary": str(M8_V1B_SUMMARY_PATH),
            "val_summary": str(VAL_SUMMARY_PATH),
            "test_summary": str(TEST_SUMMARY_PATH),
        },
        "pipeline_stages": {
            "stage_a": "baseline evidence collection from existing M8 summaries only",
            "stage_b": "xai scale attribution already summarized upstream",
            "stage_c": "size-wise aggregation over small, medium, and large",
            "stage_d": "policy mining with conservative identity fallback",
            "stage_e": "machine-readable export only",
            "stage_f": "runtime integration not executed in this phase",
            "stage_g": "controlled training and evaluation blocked pending approval",
        },
        "policy_groups": group_summaries,
        "identity_fallback_groups": ["tiny", "unknown"],
        "metric_backlog": [
            "energy_in_box",
            "background_leakage",
            "peak_alignment",
            "scale_confidence_contribution",
            "detection_error_correlation",
        ],
        "next_allowed_step": "review_m8_v1c_policy_mining_proposal_before_metric_runner",
        "training_execution_allowed": False,
        "evaluation_execution_allowed": False,
        "inference_execution_allowed": False,
        "dataset_mutation_allowed": False,
        "checkpoint_mutation_allowed": False,
    }
    write_json(OUTPUT_DIR / "policy_confidence_report.json", confidence_report)

    non_execution_manifest = {
        "phase": "M8_v1c",
        "status": "m8_v1c_policy_candidate_prepared",
        "generated_at_utc": generated_at,
        "artifacts_dir": str(OUTPUT_DIR),
        "execution_flags": {
            "training_execution_allowed": False,
            "evaluation_execution_allowed": False,
            "inference_execution_allowed": False,
            "dataset_mutation_allowed": False,
            "checkpoint_mutation_allowed": False,
        },
        "prohibited_actions_confirmed_not_run": [
            "training",
            "full_validation",
            "fresh_dataset_inference",
            "checkpoint_loading",
            "checkpoint_mutation",
            "dataset_mutation",
        ],
        "next_allowed_step": "review_m8_v1c_policy_mining_proposal_before_metric_runner",
    }
    write_json(OUTPUT_DIR / "m8_v1c_non_execution_manifest.json", non_execution_manifest)

    readme = """# M8_v1c Automated Policy Mining Preparation

This package converts existing M8 evidence into a machine-readable policy-mining proposal.
It is preparation-only and does not execute training, validation, inference, checkpoint loading, or dataset mutation.

## Files
- `scale_evidence_by_size.csv`
- `policy_candidate.yaml`
- `policy_confidence_report.json`
- `identity_fallback_cases.csv`
- `human_review_policy_template.csv`
- `m8_v1c_non_execution_manifest.json`

## Interpretation Boundary
- Evidence source is the existing dominant-scale summaries only.
- The full metric runner for energy-in-box, leakage, peak alignment, and confidence ablations is not implemented in this phase.
- Any downstream runtime integration or training remains blocked until a later reviewed phase.

## Next Allowed Step
`review_m8_v1c_policy_mining_proposal_before_metric_runner`
"""
    write_text(OUTPUT_DIR / "README.md", readme)


if __name__ == "__main__":
    main()
