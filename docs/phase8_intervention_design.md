# Phase 8 Intervention Decision Design

## Goal

Phase 8 converts the descriptive evidence from Phase 6 and the manual-review infrastructure from Phase 7 into a conservative intervention-design layer.

This phase does not:

- train any model,
- modify the training loop,
- change model architecture,
- alter YOLO wrappers,
- alter loss behavior,
- alter CAM/XAI extraction logic,
- claim that any intervention already improves performance.

## Input Artifacts

Primary inputs, if available:

- `artifacts/xai_evidence_review_chipped/representative_cases.csv`
- `artifacts/xai_evidence_review_chipped/evidence_group_summary.csv`
- `artifacts/manual_evidence_review_chipped/manual_review_summary.csv`
- `artifacts/manual_evidence_review_chipped/manual_review_summary.json`
- `artifacts/manual_evidence_review_chipped/manual_review_filled.csv`

The Phase 8 script is intentionally robust to missing files. If one or more inputs are absent, it should emit warnings and still write partial outputs and a README rather than crash without explanation.

## Output Artifacts

- `artifacts/intervention_design_chipped/intervention_decision_table.csv`
- `artifacts/intervention_design_chipped/intervention_decision_table.json`
- `artifacts/intervention_design_chipped/intervention_candidates.csv`
- `artifacts/intervention_design_chipped/intervention_candidates.json`
- `artifacts/intervention_design_chipped/no_intervention_or_insufficient_evidence.csv`
- `artifacts/intervention_design_chipped/README.md`

## Decision Taxonomy

### Evidence strength

Each evidence bucket is classified into one of:

- `strong_xai_support`
- `weak_or_mixed_xai_support`
- `insufficient_evidence`
- `do_not_intervene_yet`

The implementation should remain defensive:

- use available numeric metrics if they exist,
- avoid hard-crashing when a metric column is missing,
- degrade to weaker decisions when evidence is incomplete,
- keep manual review conservative unless it is verified as real review data.

### Intervention families

Phase 8 generates candidates only from this taxonomy:

- `DATA_AUDIT_OR_RELABEL`
- `DATA_SAMPLING_OR_CURRICULUM`
- `AUGMENTATION_REBALANCE`
- `LOSS_REWEIGHTING_PROTOTYPE`
- `SALIENCY_GUIDED_ATTENTION_PROTOTYPE`
- `NO_INTERVENTION_YET`

These families are planning labels, not implemented methods.

## Manual Review Handling

Manual-review outputs can exist in at least three states:

1. Missing
2. Demo or synthetic smoke data
3. Unknown or unverified review provenance

If `manual_review_filled.csv` contains synthetic/demo markers, Phase 8 must record:

- `manual_review_source = demo_or_synthetic`
- `manual_review_used_as_research_evidence = false`

If provenance cannot be verified, Phase 8 should still choose the conservative setting:

- `manual_review_source = unknown`
- `manual_review_used_as_research_evidence = false`

This matters because Phase 8 is allowed to validate code paths with demo review data, but it is not allowed to use demo review rows as scientific evidence.

## CLI

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/design_phase8_interventions.py \
  --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv \
  --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv \
  --manual-review-summary artifacts/manual_evidence_review_chipped/manual_review_summary.csv \
  --manual-review-filled artifacts/manual_evidence_review_chipped/manual_review_filled.csv \
  --output artifacts/intervention_design_chipped \
  --focus-class Chipped
```

Then compile-check:

```bash
.venv/bin/python -m compileall src scripts
```

## How To Read The Outputs

### `intervention_decision_table.csv/json`

These files summarize each evidence bucket and assign:

- evidence strength,
- manual-review status,
- intervention family,
- whether it is safe to prototype,
- rationale,
- the next evidence still required.

### `intervention_candidates.csv/json`

These are Phase 9 planning candidates only. They should be interpreted as hypothesis-level next steps, not as validated improvements.

### `no_intervention_or_insufficient_evidence.csv`

This file is intentionally important. It records the buckets that should not yet trigger an intervention because the evidence is weak, mixed, incomplete, or blocked by uncertain manual review provenance.

## Limitations

- Phase 8 depends entirely on upstream evidence quality.
- Phase 8 does not resolve causal interpretation.
- Phase 8 does not prove an intervention will help.
- If manual review is demo/synthetic or unknown, the decision layer must remain conservative.
- Mixed or borderline buckets should prefer more evidence over premature intervention design.

## Gate To Phase 9

Phase 9 should begin only when all of the following are true:

- manual review has been completed with real, non-demo annotations,
- repeated patterns appear across multiple representative buckets,
- the proposed intervention family is supported by descriptive evidence rather than one-off cases,
- label ambiguity has been ruled out where relevant,
- the team accepts that Phase 9 is still an experimental prototype stage, not proof of causal correctness.
