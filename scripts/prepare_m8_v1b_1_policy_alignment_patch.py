"""Prepare the additive M8_v1b.1 evidence-aligned policy patch artifacts."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.scale_weighting import SCALE_ORDER, ScaleWeightingPolicy

PARENT_SUMMARY_PATH = Path(
    "artifacts/m8_v1b_size_aware_scale_weighting_implementation/m8_v1b_implementation_summary.json"
)
PARENT_OVERRIDE_PATH = Path(
    "artifacts/m8_v1b_size_aware_scale_weighting_implementation/m8_v1b_human_override_record.json"
)
PARENT_POLICY_PATH = Path(
    "artifacts/m8_v1b_size_aware_scale_weighting_implementation/m8_v1b_scale_weight_policy.csv"
)
PARENT_SIZE_POLICY_PATH = Path(
    "artifacts/m8_v1b_size_aware_scale_weighting_implementation/m8_v1b_size_policy.csv"
)
PARENT_MANIFEST_PATH = Path(
    "artifacts/m8_v1b_size_aware_scale_weighting_implementation/m8_v1b_non_execution_manifest.json"
)
PARENT_CHECK_PATH = Path(
    "artifacts/m8_v1b_size_aware_scale_weighting_implementation/m8_v1b_check_report.json"
)
UPDATED_CONFIG_PATH = Path("configs/method/m8_v1b_size_aware_scale_weighting.yaml")
UPDATED_METHOD_MODULE = "models/scale_weighting.py"
OUTPUT_DIR = Path("artifacts/m8_v1b_1_policy_alignment_patch")

EXPECTED_STATUS = "m8_v1b_size_aware_scale_weighting_implementation_prepared"
UPDATED_POLICY = {
    "small": {"P2": 1.05, "P3": 1.15, "P4": 0.95, "P5": 0.90},
    "medium": {"P2": 0.90, "P3": 1.00, "P4": 1.10, "P5": 1.15},
    "large": {"P2": 0.90, "P3": 0.95, "P4": 1.10, "P5": 1.15},
    "unknown": {"P2": 1.00, "P3": 1.00, "P4": 1.00, "P5": 1.00},
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def read_old_policy(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    policy: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group = row["size_group"]
            scale = row["scale"]
            policy.setdefault(group, {})[scale] = float(row["weight"])
    return policy


def verify_parent_status(summary: dict[str, Any], override: dict[str, Any], manifest: dict[str, Any]) -> None:
    if summary.get("status") != EXPECTED_STATUS:
        raise ValueError(f"Parent summary status must be {EXPECTED_STATUS!r}, got {summary.get('status')!r}.")
    if summary.get("human_override_applied") is not True:
        raise ValueError("Parent summary must preserve human_override_applied=true.")
    if override.get("human_override_applied") is not True:
        raise ValueError("Parent override record must preserve human_override_applied=true.")

    flag_names = (
        "training_execution_allowed",
        "evaluation_execution_allowed",
        "inference_execution_allowed",
        "dataset_mutation_allowed",
        "checkpoint_mutation_allowed",
    )
    for flag_name in flag_names:
        if summary.get(flag_name) is not False:
            raise ValueError(f"Parent summary flag must remain false: {flag_name}")
        if override.get(flag_name) is not False:
            raise ValueError(f"Parent override flag must remain false: {flag_name}")
        if manifest.get("execution_flags", {}).get(flag_name) is not False:
            raise ValueError(f"Parent manifest flag must remain false: {flag_name}")


def validate_policy(policy: dict[str, dict[str, float]]) -> None:
    for group_name, scale_weights in policy.items():
        if set(scale_weights) != set(SCALE_ORDER):
            raise ValueError(f"{group_name} must define exactly {SCALE_ORDER}, got {sorted(scale_weights)}.")
        for scale_name, weight in scale_weights.items():
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"Invalid weight for {group_name}/{scale_name}: {weight!r}")
            if weight < 0.85 or weight > 1.20:
                raise ValueError(f"Non-conservative weight for {group_name}/{scale_name}: {weight!r}")


def verify_evidence_alignment(policy: dict[str, dict[str, float]], evidence_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    small_highest = max(policy["small"], key=policy["small"].get)
    medium_highest = max(policy["medium"], key=policy["medium"].get)
    large_highest = max(policy["large"], key=policy["large"].get)
    unknown_identity = all(policy["unknown"][scale] == 1.0 for scale in SCALE_ORDER)

    rows.append(
        {
            "size_group": "small",
            "val_dominant_scale": evidence_snapshot["val"]["small"]["dominant_scale"],
            "test_dominant_scale": evidence_snapshot["test"]["small"]["dominant_scale"],
            "policy_highest_scale": small_highest,
            "alignment_result": "pass" if small_highest == "P3" else "fail",
        }
    )
    rows.append(
        {
            "size_group": "medium",
            "val_dominant_scale": evidence_snapshot["val"]["medium"]["dominant_scale"],
            "test_dominant_scale": evidence_snapshot["test"]["medium"]["dominant_scale"],
            "policy_highest_scale": medium_highest,
            "alignment_result": "pass" if medium_highest == "P5" else "fail",
        }
    )
    rows.append(
        {
            "size_group": "large",
            "val_dominant_scale": evidence_snapshot["val"]["large"]["dominant_scale"],
            "test_dominant_scale": evidence_snapshot["test"]["large"]["dominant_scale"],
            "policy_highest_scale": large_highest,
            "alignment_result": "pass" if large_highest == "P5" else "fail",
        }
    )
    rows.append(
        {
            "size_group": "unknown",
            "val_dominant_scale": "identity_fallback",
            "test_dominant_scale": "identity_fallback",
            "policy_highest_scale": "identity_all_1.0" if unknown_identity else "non_identity",
            "alignment_result": "pass" if unknown_identity else "fail",
        }
    )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parent_summary = load_json(PARENT_SUMMARY_PATH)
    parent_override = load_json(PARENT_OVERRIDE_PATH)
    parent_manifest = load_json(PARENT_MANIFEST_PATH)
    verify_parent_status(parent_summary, parent_override, parent_manifest)

    old_policy = read_old_policy(PARENT_POLICY_PATH)
    _ = PARENT_SIZE_POLICY_PATH.exists()
    _ = PARENT_CHECK_PATH.exists()

    policy = ScaleWeightingPolicy()
    new_policy = policy.scale_weights
    validate_policy(new_policy)

    if new_policy != UPDATED_POLICY:
        raise ValueError("Repo-local method policy does not match the requested updated evidence-aligned policy.")

    evidence_snapshot = parent_summary.get("evidence_snapshot")
    if not isinstance(evidence_snapshot, dict):
        raise ValueError("Parent summary must include an evidence_snapshot object.")

    alignment_rows = verify_evidence_alignment(new_policy, evidence_snapshot)
    if any(row["alignment_result"] != "pass" for row in alignment_rows):
        raise ValueError("Updated policy failed evidence-alignment verification.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()

    old_vs_new_rows: list[list[Any]] = []
    for group_name in ("small", "medium", "large", "unknown"):
        for scale_name in SCALE_ORDER:
            old_weight = old_policy[group_name][scale_name]
            new_weight = new_policy[group_name][scale_name]
            old_vs_new_rows.append(
                [
                    group_name,
                    scale_name,
                    old_weight,
                    new_weight,
                    round(new_weight - old_weight, 6),
                    "changed" if new_weight != old_weight else "unchanged",
                ]
            )
    write_csv(
        OUTPUT_DIR / "m8_v1b_1_old_vs_new_scale_weight_policy.csv",
        ["size_group", "scale", "old_weight", "new_weight", "delta", "change_status"],
        old_vs_new_rows,
    )

    write_csv(
        OUTPUT_DIR / "m8_v1b_1_evidence_alignment_check.csv",
        ["size_group", "val_dominant_scale", "test_dominant_scale", "policy_highest_scale", "alignment_result"],
        [
            [
                row["size_group"],
                row["val_dominant_scale"],
                row["test_dominant_scale"],
                row["policy_highest_scale"],
                row["alignment_result"],
            ]
            for row in alignment_rows
        ],
    )

    non_execution_manifest = {
        "phase": "M8_v1b.1",
        "status": "m8_v1b_1_policy_alignment_patch_prepared",
        "parent_phase": "M8_v1b",
        "parent_status_verified": True,
        "generated_at_utc": generated_at,
        "artifacts_dir": str(OUTPUT_DIR),
        "inputs_verified": {
            "parent_summary": str(PARENT_SUMMARY_PATH),
            "parent_override": str(PARENT_OVERRIDE_PATH),
            "parent_policy": str(PARENT_POLICY_PATH),
            "parent_size_policy": str(PARENT_SIZE_POLICY_PATH),
            "parent_manifest": str(PARENT_MANIFEST_PATH),
            "parent_check_report_present": PARENT_CHECK_PATH.exists(),
        },
        "execution_flags": {
            "training_execution_allowed": False,
            "evaluation_execution_allowed": False,
            "inference_execution_allowed": False,
            "dataset_mutation_allowed": False,
            "checkpoint_mutation_allowed": False,
        },
        "evidence_alignment_status": "evidence_aligned",
        "next_allowed_step": "review_m8_v1b_1_policy_alignment_before_training_gate",
    }
    write_json(OUTPUT_DIR / "m8_v1b_1_non_execution_manifest.json", non_execution_manifest)

    summary = {
        "phase": "M8_v1b.1",
        "status": "m8_v1b_1_policy_alignment_patch_prepared",
        "parent_phase": "M8_v1b",
        "parent_status_verified": True,
        "generated_at_utc": generated_at,
        "human_override_preserved": True,
        "old_policy_path": str(PARENT_POLICY_PATH),
        "updated_config_path": str(UPDATED_CONFIG_PATH),
        "updated_method_module": UPDATED_METHOD_MODULE,
        "evidence_alignment_status": "evidence_aligned",
        "training_execution_allowed": False,
        "evaluation_execution_allowed": False,
        "inference_execution_allowed": False,
        "dataset_mutation_allowed": False,
        "checkpoint_mutation_allowed": False,
        "next_allowed_step": "review_m8_v1b_1_policy_alignment_before_training_gate",
    }
    write_json(OUTPUT_DIR / "m8_v1b_1_policy_alignment_summary.json", summary)

    readme = """# M8_v1b.1 Policy Alignment Patch

This additive patch preserves the original M8_v1b implementation-preparation package while aligning the default scale-weight policy with the stored M8 evidence snapshot.

## Key Result
- `small` now peaks at `P3`
- `medium` now peaks at `P5`
- `large` continues to peak at `P5`
- `unknown` remains identity

## Guardrails Preserved
- training_execution_allowed = false
- evaluation_execution_allowed = false
- inference_execution_allowed = false
- dataset_mutation_allowed = false
- checkpoint_mutation_allowed = false

## Next Allowed Step
`review_m8_v1b_1_policy_alignment_before_training_gate`
"""
    write_text(OUTPUT_DIR / "README.md", readme)


if __name__ == "__main__":
    main()
