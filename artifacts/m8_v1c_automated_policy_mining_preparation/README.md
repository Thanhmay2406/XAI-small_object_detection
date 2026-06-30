# M8_v1c Automated Policy Mining Preparation

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
