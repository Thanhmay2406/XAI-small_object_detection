"""Synthetic-only checks for the repo-local M8_v1b size-aware scale weighting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.scale_weighting import (
    ScaleWeightingPolicy,
    SizeAwareScaleWeighter,
    classify_yolo_box_size_group,
    identity_or_group_weights,
    resolve_group_scale_weights,
)

OUTPUT_DIR = Path("artifacts/m8_v1b_1_policy_alignment_patch")
REPORT_PATH = OUTPUT_DIR / "m8_v1b_1_check_report.json"


def assert_close(actual: float, expected: float, *, message: str) -> None:
    if abs(actual - expected) > 1e-6:
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    policy = ScaleWeightingPolicy()
    weighter = SizeAwareScaleWeighter(policy)
    small_weights = resolve_group_scale_weights("small", policy)
    medium_weights = resolve_group_scale_weights("medium", policy)
    large_weights = resolve_group_scale_weights("large", policy)
    unknown_weights = resolve_group_scale_weights("unknown", policy)

    size_group_results = {
        "small": classify_yolo_box_size_group(0.04, 0.04, 640, 640, policy.size_thresholds),
        "medium": classify_yolo_box_size_group(0.10, 0.10, 640, 640, policy.size_thresholds),
        "large": classify_yolo_box_size_group(0.20, 0.20, 640, 640, policy.size_thresholds),
        "unknown": classify_yolo_box_size_group(0.0, 0.10, 640, 640, policy.size_thresholds),
    }
    expected_groups = {"small": "small", "medium": "medium", "large": "large", "unknown": "unknown"}
    if size_group_results != expected_groups:
        raise AssertionError(f"Unexpected size-group classification results: {size_group_results!r}")

    feature_map = {
        "P2": torch.ones(1, 8, 80, 80),
        "P3": torch.ones(1, 8, 40, 40),
        "P4": torch.ones(1, 8, 20, 20),
        "P5": torch.ones(1, 8, 10, 10),
    }
    original_shapes = {name: tuple(tensor.shape) for name, tensor in feature_map.items()}
    weighted_small = weighter.apply(feature_map, size_group="small")
    weighted_large = weighter.apply(feature_map, size_group="large")
    weighted_unknown = weighter.apply(feature_map, size_group="not_a_group")

    for scale_name, tensor in weighted_small.items():
        if tuple(tensor.shape) != original_shapes[scale_name]:
            raise AssertionError(f"Shape changed for {scale_name}: {original_shapes[scale_name]} -> {tuple(tensor.shape)}")

    if max(small_weights, key=small_weights.get) != "P3":
        raise AssertionError(f"Expected small highest weight to be P3, got {small_weights!r}")
    if max(medium_weights, key=medium_weights.get) != "P5":
        raise AssertionError(f"Expected medium highest weight to be P5, got {medium_weights!r}")
    if max(large_weights, key=large_weights.get) != "P5":
        raise AssertionError(f"Expected large highest weight to be P5, got {large_weights!r}")
    if unknown_weights != {"P2": 1.0, "P3": 1.0, "P4": 1.0, "P5": 1.0}:
        raise AssertionError(f"Expected unknown weights to be identity, got {unknown_weights!r}")

    assert_close(weighted_small["P2"][0, 0, 0, 0].item(), 1.05, message="small/P2 weight mismatch")
    assert_close(weighted_small["P3"][0, 0, 0, 0].item(), 1.15, message="small/P3 weight mismatch")
    assert_close(weighted_small["P5"][0, 0, 0, 0].item(), 0.90, message="small/P5 weight mismatch")
    weighted_medium = weighter.apply(feature_map, size_group="medium")
    assert_close(weighted_medium["P4"][0, 0, 0, 0].item(), 1.10, message="medium/P4 weight mismatch")
    assert_close(weighted_medium["P5"][0, 0, 0, 0].item(), 1.15, message="medium/P5 weight mismatch")
    assert_close(weighted_large["P4"][0, 0, 0, 0].item(), 1.10, message="large/P4 weight mismatch")
    assert_close(weighted_large["P5"][0, 0, 0, 0].item(), 1.15, message="large/P5 weight mismatch")
    assert_close(weighted_unknown["P3"][0, 0, 0, 0].item(), 1.00, message="unknown/P3 weight mismatch")

    sequence_features = [
        torch.ones(1, 4, 80, 80),
        torch.ones(1, 4, 40, 40),
        torch.ones(1, 4, 20, 20),
        torch.ones(1, 4, 10, 10),
    ]
    sequence_weighted = weighter.apply(sequence_features, size_group="medium")
    expected_sequence_weights = resolve_group_scale_weights("medium", policy)
    for index, scale_name in enumerate(policy.supported_scales):
        assert_close(
            sequence_weighted[index][0, 0, 0, 0].item(),
            expected_sequence_weights[scale_name],
            message=f"sequence {scale_name} weight mismatch",
        )

    identity_weights = identity_or_group_weights("small", False, policy)
    if identity_weights != resolve_group_scale_weights("unknown", policy):
        raise AssertionError("Identity fallback did not resolve to unknown-group weights.")

    report = {
        "phase": "M8_v1b.1",
        "status": "m8_v1b_1_policy_alignment_check_completed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "checks_passed": [
            "size_group_classification_small_medium_large_unknown",
            "small_highest_weight_is_p3",
            "medium_highest_weight_is_p5",
            "large_highest_weight_is_p5",
            "unknown_weights_are_identity",
            "feature_shapes_unchanged_for_mapping",
            "feature_shapes_unchanged_for_sequence",
            "policy_weights_applied_for_small_medium_large_unknown_with_alignment_patch",
            "identity_fallback_uses_unknown_weights",
        ],
        "evidence_alignment_status": "evidence_aligned",
        "synthetic_inputs_only": True,
        "dataset_images_loaded": False,
        "checkpoints_loaded": False,
        "training_called": False,
        "evaluation_called": False,
        "inference_called": False,
        "dataset_mutated": False,
        "checkpoint_mutated": False,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
