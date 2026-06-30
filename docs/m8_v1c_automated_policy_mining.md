# M8_v1c Automated XAI-guided Size-aware Scale Policy Mining

## Objective
Define the next conservative step after `M8_v1b`: convert XAI scale evidence into a machine-readable policy proposal instead of relying on manual visual judgment alone.

This phase is still preparation-only:
- `training_execution_allowed = false`
- `evaluation_execution_allowed = false`
- `inference_execution_allowed = false`
- `dataset_mutation_allowed = false`
- `checkpoint_mutation_allowed = false`

## Method Positioning
`M8_v1b` introduced a manually chosen, conservative size-aware weighting policy.
`M8_v1c` upgrades that direction into an evidence-to-policy pipeline:

1. Read baseline evidence
2. Aggregate evidence by object-size group
3. Score candidate scales per group
4. Apply conservative thresholds
5. Emit a machine-readable policy
6. Fall back to identity when evidence is weak or ambiguous

## Target Research Claim
Instead of saying:

`Based on XAI, we manually chose weights for P3/P4/P5.`

`M8_v1c` aims to support the stronger claim:

`We propose an automated pipeline that mines XAI evidence to infer size-aware scale-weighting policy candidates, with identity fallback when confidence is insufficient.`

## Pipeline Stages

### Stage A - Baseline evidence collection
- Use existing baseline outputs and matched GT/prediction context
- Preserve non-execution guardrails in preparation phases

### Stage B - XAI scale attribution
- Extract or reuse scale-specific XAI evidence for `P3`, `P4`, and `P5`
- Keep per-object attribution records rather than only final conclusions

### Stage C - Size-wise aggregation
- Partition objects into `small`, `medium`, `large`
- Keep `tiny` and `unknown` separate so they can fail closed

### Stage D - Policy mining
- Infer `preferred_scale = argmax EvidenceScore(size_group, scale)`
- Require confidence checks before auto-proposing a non-identity policy

### Stage E - Policy export
- Save policy YAML/JSON
- Save confidence report, fallback cases, and human-review template

### Stage F - Runtime integration
- Remains a future reviewed phase only
- No runtime hook or trainer wiring is executed here

### Stage G - Controlled training/eval
- Explicitly blocked pending separate approval

## Metric Backlog for the Full Runner
The full metric runner is intentionally not implemented in this preparation phase.
It should eventually score at least:

1. `energy_in_box`
2. `background_leakage`
3. `peak_alignment`
4. `scale_confidence_contribution`
5. `detection_error_correlation`

These metrics should be aggregated per size group and per scale before policy inference.

## Conservative Identity Fallback
Identity fallback is a first-class rule, not an error case.

Suggested decision rule:

```python
if best_scale_score - second_best_scale_score < threshold:
    policy[size_group] = "identity"
else:
    policy[size_group] = best_scale
```

Additional conservative gates should also be allowed, such as:
- minimum object count
- split consistency between validation and test exploratory evidence
- manual review required before any trainer-time integration

## Current Preparation Outputs
The preparation script writes:

1. `scale_evidence_by_size.csv`
2. `policy_candidate.yaml`
3. `policy_confidence_report.json`
4. `identity_fallback_cases.csv`
5. `human_review_policy_template.csv`

All outputs in this phase are proposal artifacts only.
They do not authorize training or claim measured downstream gains.

## Current Evidence Boundary
The initial `M8_v1c` preparation package reuses the existing `M8.0/M8.1` dominant-scale summaries as a seed signal.
That means the first package is honest about its limitation:

- it is a bridge from manual policy selection to metric-based policy mining
- it is not yet the full per-object metric runner
- it remains suitable for review, schema design, and next-step implementation planning

## Next Allowed Step
`review_m8_v1c_policy_mining_proposal_before_metric_runner`
