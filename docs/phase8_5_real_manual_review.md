# Phase 8.5 Real Manual Review Completion Support

## Goal

Phase 8.5 prepares the project to replace demo or smoke manual-review content with real manual review.

This phase does not:

- train a model,
- change any training config,
- modify YOLO wrappers,
- modify model architecture,
- modify any real loss,
- modify CAM or XAI extraction logic,
- invent manual labels on behalf of the reviewer.

## Why Demo Review Cannot Be Used

The current `artifacts/manual_evidence_review_chipped/manual_review_filled.csv` contains synthetic smoke-test annotations.

That means:

- it can validate code paths,
- it can validate the Phase 7 summary branch,
- it can validate the Phase 8 decision-design branch,
- but it cannot serve as research evidence.

As long as demo or synthetic markers remain in the review file, the project must keep:

- `manual_review_source = demo_or_synthetic`
- `manual_review_used_as_research_evidence = false`

## Files To Use For Real Review

Primary files:

- `artifacts/manual_evidence_review_chipped/manual_review_real_template.csv`
- `artifacts/manual_evidence_review_chipped/manual_review_guide.md`
- `artifacts/xai_evidence_review_chipped/review_notes_template.csv`
- `artifacts/xai_evidence_review_chipped/representative_cases.csv`
- `artifacts/xai_evidence_review_chipped/evidence_group_summary.csv`

Visual evidence paths referenced inside the CSV:

- `overlay_path`
- `crop_path`

## How To Fill Each Review Column

Reviewer-entered columns:

- `visual_evidence_quality`
  Use one of: `clear`, `weak`, `ambiguous`, `not_visible`
- `saliency_alignment`
  Use one of: `aligned`, `partially_aligned`, `misaligned`, `absent`, `uncertain`
- `failure_cause_hypothesis`
  Use one of: `weak_visual_cue`, `small_object`, `background_confusion`, `localization_shift`, `class_confusion`, `possible_label_noise`, `cam_method_uncertain`, `other`
- `label_quality`
  Use one of: `ok`, `questionable`, `likely_noise`, `uncertain`
- `recommended_action`
  Use one of: `hard_sample_weighting`, `chipped_focused_augmentation`, `background_negative_mining`, `label_review`, `cross_method_xai_check`, `no_action`
- `reviewer_notes`
  Write short descriptive notes only. Do not write causal claims such as “the model definitely used region X”.

## If A Case Is Uncertain

When a case is genuinely unclear:

- prefer `uncertain` or `ambiguous` over forcing a strong label,
- explain the ambiguity in `reviewer_notes`,
- use `cam_method_uncertain` when the XAI pattern itself is not trustworthy enough,
- use `label_quality = questionable` or `likely_noise` only when the ambiguity really appears label-related.

The goal is honest descriptive review, not aggressive completion.

## Priority Buckets

Finish these first:

- `localization_misaligned_evidence`
- `near_threshold_high_evidence`

Then continue with:

- `fn_low_evidence`
- `fn_misaligned_evidence`
- `fp_background_like_evidence`
- `near_threshold_low_evidence`
- `tp_high_evidence`

These first two buckets matter most because Phase 8 currently exported cautious intervention candidates tied to them, but those candidates are still blocked by synthetic manual review provenance.

## Validation Step

Before rerunning Phase 7 or Phase 8, validate the manual review file:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_manual_review_real.py \
  --manual-review artifacts/manual_evidence_review_chipped/manual_review_filled.csv \
  --output artifacts/manual_evidence_review_chipped/manual_review_real_validation.json
```

This validator checks:

- the file exists,
- demo or synthetic markers are gone,
- required review columns are present,
- categorical values stay inside the Phase 7 schema,
- enough rows are actually completed,
- the file is suitable for research-evidence use.

## Rerun Phase 7 Summary After Real Review

Only after validation passes:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_manual_evidence_review.py \
  --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv \
  --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv \
  --output artifacts/manual_evidence_review_chipped \
  --focus-class Chipped
```

## Rerun Phase 8 Decision Design After Real Review

Then rerun:

```bash
PYTHONPATH=src .venv/bin/python scripts/design_phase8_interventions.py \
  --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv \
  --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv \
  --manual-review-summary artifacts/manual_evidence_review_chipped/manual_review_summary.csv \
  --manual-review-filled artifacts/manual_evidence_review_chipped/manual_review_filled.csv \
  --output artifacts/intervention_design_chipped \
  --focus-class Chipped
```

## Gate To Phase 9

Phase 9 should remain blocked until all of the following are true:

- `manual_review_filled.csv` no longer contains demo or synthetic content,
- real review validation passes,
- Phase 7 summary is regenerated from real review,
- Phase 8 decision design is regenerated from real review,
- the updated decision table still supports a repeated, interpretable pattern across multiple rows,
- the team accepts that any Phase 9 prototype is still a research hypothesis, not proof of model improvement.
