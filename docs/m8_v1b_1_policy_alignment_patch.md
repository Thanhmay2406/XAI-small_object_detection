# M8_v1b.1 Policy Alignment Patch

## Why M8_v1b.1 Is Needed
The original M8_v1b implementation package was safe and conservative, but its default scale-weight policy was not fully aligned with the stored M8 evidence snapshot.
That mismatch matters because the prepared repo-local policy should reflect the evidence it cites before any later training gate is even considered.

## Previous Policy Mismatch
The earlier M8_v1b defaults had two main problems:
- `small` assigned the highest weight to `P2`, even though both val and test evidence favored `P3`
- `medium` downweighted `P5`, even though both val and test evidence favored `P5`

`large` was already reasonably aligned because `P5` remained the highest weight there.

## Evidence-Aligned Updated Policy

| Size group | P2 | P3 | P4 | P5 |
| --- | ---: | ---: | ---: | ---: |
| small | 1.05 | 1.15 | 0.95 | 0.90 |
| medium | 0.90 | 1.00 | 1.10 | 1.15 |
| large | 0.90 | 0.95 | 1.10 | 1.15 |
| unknown | 1.00 | 1.00 | 1.00 | 1.00 |

This remains intentionally conservative:
- no weight below `0.85`
- no weight above `1.20`
- `unknown` stays near-identity and exactly identity by default

## What Changed
- Updated the repo-local default policy in `models/scale_weighting.py`
- Updated the repo-local YAML in `configs/method/m8_v1b_size_aware_scale_weighting.yaml`
- Extended the synthetic checker to verify evidence alignment and write a patch-specific report
- Added additive M8_v1b.1 patch artifacts under `artifacts/m8_v1b_1_policy_alignment_patch/`

## What Did Not Change
- size-threshold logic
- shape-safety behavior
- finite/positive weight validation
- identity fallback behavior for unknown or invalid size metadata
- trainer integration boundaries
- execution and mutation guardrails

## Guardrails Preserved
- `training_execution_allowed = false`
- `evaluation_execution_allowed = false`
- `inference_execution_allowed = false`
- `dataset_mutation_allowed = false`
- `checkpoint_mutation_allowed = false`

## Limitations
- No training was run
- No validation or prediction was run
- No checkpoint was loaded
- No runtime gain is claimed yet
- This remains an implementation-preparation patch, not a training authorization

## Next Allowed Step
`review_m8_v1b_1_policy_alignment_before_training_gate`
