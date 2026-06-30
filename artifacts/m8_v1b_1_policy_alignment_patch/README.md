# M8_v1b.1 Policy Alignment Patch

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
