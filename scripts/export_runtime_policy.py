"""Export a mined scale policy candidate into a runtime-ready config YAML.

This script is intentionally narrow:
- it reads an analysis-generated policy candidate YAML
- it normalizes missing groups to conservative identity weights
- it writes a runtime config compatible with models/m8_v1c_runtime.py

It does not run training, evaluation, inference, or mutate checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SCALES = ("P2", "P3", "P4", "P5")
SUPPORTED_GROUPS = ("small", "medium", "large", "unknown")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a mined scale policy candidate to a runtime-ready config YAML."
    )
    parser.add_argument(
        "--policy-candidate",
        type=Path,
        required=True,
        help="Path to m8_v1c_metric_policy_candidate.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/method/runtime_scale_policy.yaml"),
        help="Output runtime config YAML path.",
    )
    parser.add_argument(
        "--small-max-pixel-area",
        type=float,
        default=1024.0,
        help="Runtime threshold for mapping image-level GT groups to small.",
    )
    parser.add_argument(
        "--medium-max-pixel-area",
        type=float,
        default=9216.0,
        help="Runtime threshold for mapping image-level GT groups to medium.",
    )
    return parser


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def identity_weights() -> dict[str, float]:
    return {scale: 1.0 for scale in SUPPORTED_SCALES}


def validate_weight_delta(group_name: str, weight_delta: Any) -> dict[str, float]:
    if not isinstance(weight_delta, dict):
        raise ValueError(f"{group_name} weight_delta must be a mapping.")
    resolved = {str(scale): float(weight) for scale, weight in weight_delta.items()}
    if set(resolved) != set(SUPPORTED_SCALES):
        raise ValueError(
            f"{group_name} weight_delta must define exactly {list(SUPPORTED_SCALES)}, got {sorted(resolved)}."
        )
    return resolved


def normalize_group(group_name: str, raw_group: Any) -> dict[str, Any]:
    if not isinstance(raw_group, dict):
        return {"mode": "identity", "weight_delta": identity_weights()}

    mode = str(raw_group.get("mode", "identity"))
    if mode == "preferred_scale":
        preferred_scale = raw_group.get("preferred_scale")
        if preferred_scale is None:
            raise ValueError(f"{group_name} preferred_scale is required when mode=preferred_scale.")
        preferred_scale = str(preferred_scale)
        if preferred_scale not in SUPPORTED_SCALES:
            raise ValueError(
                f"{group_name} preferred_scale must be one of {list(SUPPORTED_SCALES)}, got {preferred_scale!r}."
            )
        weight_delta = validate_weight_delta(group_name, raw_group.get("weight_delta"))
        return {
            "mode": "preferred_scale",
            "preferred_scale": preferred_scale,
            "weight_delta": weight_delta,
        }

    return {"mode": "identity", "weight_delta": identity_weights()}


def export_runtime_payload(
    candidate: dict[str, Any],
    *,
    policy_candidate_path: Path,
    small_max_pixel_area: float,
    medium_max_pixel_area: float,
) -> dict[str, Any]:
    size_aware_policy = candidate.get("size_aware_policy")
    if not isinstance(size_aware_policy, dict):
        raise ValueError("policy candidate must contain a size_aware_policy mapping.")

    runtime_policy = {
        group_name: normalize_group(group_name, size_aware_policy.get(group_name))
        for group_name in SUPPORTED_GROUPS
    }

    return {
        "method_name": "m8_v1c_instance_aware_spatial_scale_weighting",
        "method": "m8_v1c_instance_aware_spatial_scale_weighting",
        "enabled": True,
        "implementation_phase": "runtime_integration_ready",
        "runtime_application_mode": "instance_aware_spatial",
        "policy_source_artifact": str(policy_candidate_path.resolve()),
        "supported_scales": list(SUPPORTED_SCALES),
        "feature_levels": list(SUPPORTED_SCALES),
        "size_thresholds": {
            "small_max_pixel_area": float(small_max_pixel_area),
            "medium_max_pixel_area": float(medium_max_pixel_area),
        },
        "size_aware_policy": runtime_policy,
        "scale_weights": {
            group_name: dict(group_payload["weight_delta"])
            for group_name, group_payload in runtime_policy.items()
        },
        "guardrails": {
            "training_execution_allowed": False,
            "evaluation_execution_allowed": False,
            "inference_execution_allowed": False,
            "dataset_mutation_allowed": False,
            "checkpoint_mutation_allowed": False,
        },
        "notes": [
            "This runtime config was exported from an M8_v1c XAI metric policy candidate artifact.",
            "Tiny groups are intentionally not mapped directly at runtime and therefore fail closed through unknown.",
            "Identity fallback is preserved for missing, ambiguous, or under-reviewed policy groups.",
        ],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    candidate = load_yaml_mapping(args.policy_candidate)
    runtime_payload = export_runtime_payload(
        candidate,
        policy_candidate_path=args.policy_candidate,
        small_max_pixel_area=args.small_max_pixel_area,
        medium_max_pixel_area=args.medium_max_pixel_area,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime_payload, handle, sort_keys=False)


if __name__ == "__main__":
    main()
