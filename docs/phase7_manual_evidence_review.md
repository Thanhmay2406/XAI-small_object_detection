# Phase 7 Manual Evidence Review and Intervention Design

## Goal

Phase 7 standardizes manual review for the Phase 6 representative `Chipped` evidence cases and prepares an intervention-design decision gate for Phase 8.

This phase does not:

- train a new model,
- add an XAI loss,
- modify the YOLO architecture,
- implement intervention training,
- claim causal conclusions from post-hoc XAI evidence.

## Inputs

- `artifacts/xai_evidence_review_chipped/representative_cases.csv`
- `artifacts/xai_evidence_review_chipped/review_notes_template.csv`
- `artifacts/xai_evidence_review_chipped/evidence_group_summary.csv`
- `artifacts/xai_evidence_chipped/overlays/`
- `artifacts/xai_evidence_chipped/contact_sheets/`

Optional input:

- `artifacts/manual_evidence_review_chipped/manual_review_filled.csv`

## Outputs

- `artifacts/manual_evidence_review_chipped/manual_review_template.csv`
- `artifacts/manual_evidence_review_chipped/manual_review_guide.md`
- `artifacts/manual_evidence_review_chipped/intervention_decision_table.md`
- `artifacts/manual_evidence_review_chipped/README.md`

Optional outputs when a filled review file is present:

- `artifacts/manual_evidence_review_chipped/manual_review_summary.csv`
- `artifacts/manual_evidence_review_chipped/manual_review_summary.json`

## Implementation

Source additions:

- `src/xai_evidence_sod/xai/manual_review.py`
- `scripts/prepare_manual_evidence_review.py`

Command used:

```bash
.venv/bin/python -m compileall src scripts
PYTHONPATH=src .venv/bin/python scripts/prepare_manual_evidence_review.py \
  --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv \
  --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv \
  --output artifacts/manual_evidence_review_chipped \
  --focus-class Chipped
```

Current run status:

- the Phase 7 preparation command completed successfully
- `manual_review_template.csv`, `manual_review_guide.md`, `intervention_decision_table.md`, and `README.md` were exported
- no `manual_review_summary.*` files were created in this run because `manual_review_filled.csv` was not present yet

## Smoke Test Demo Path

To exercise the summary branch without doing real manual annotation, use the synthetic smoke-test helper:

```bash
.venv/bin/python scripts/create_manual_review_smoke_demo.py \
  --template artifacts/manual_evidence_review_chipped/manual_review_template.csv \
  --output artifacts/manual_evidence_review_chipped/manual_review_filled.csv \
  --rows 6

PYTHONPATH=src .venv/bin/python scripts/prepare_manual_evidence_review.py \
  --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv \
  --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv \
  --output artifacts/manual_evidence_review_chipped \
  --focus-class Chipped
```

Important boundary:

- `manual_review_filled.csv` created by `scripts/create_manual_review_smoke_demo.py` is demo/synthetic content only
- `manual_review_summary.csv` and `manual_review_summary.json` produced from that file are smoke-test validation artifacts only
- those demo files must not be used as real manual-review evidence or research conclusions

## Manual Review Workflow

Recommended procedure:

1. Open `manual_review_template.csv`.
2. For each row, inspect the `overlay_path` first, then the `crop_path`.
3. Record descriptive judgments about visibility, saliency alignment, likely failure pattern, and intervention relevance.
4. Use controlled values where possible so later aggregation is consistent.
5. Keep `reviewer_notes` for uncertainty, edge cases, or hypotheses that need nuance.
6. Avoid statements that imply the highlighted region caused the prediction.

Suggested order:

- review `tp_high_evidence` first to calibrate what a comparatively strong evidence case looks like in this sample
- review `fn_low_evidence` and `fn_misaligned_evidence` next to distinguish absent evidence from displaced evidence
- review `fp_background_like_evidence` and `localization_misaligned_evidence` next to spot repeated background or shift patterns
- finish with `near_threshold_high_evidence` and `near_threshold_low_evidence` to test whether Phase 6 borderline rows form a meaningful split

## Review Schema

`manual_review_template.csv` includes these columns:

- `case_id`
- `image_path`
- `case_type`
- `bucket`
- `error_type`
- `tags`
- `energy_in_gt_box`
- `energy_in_pred_box`
- `peak_inside_gt_box`
- `saliency_concentration`
- `overlay_path`
- `crop_path`
- `visual_evidence_quality`
- `saliency_alignment`
- `failure_cause_hypothesis`
- `label_quality`
- `recommended_action`
- `reviewer_notes`

Controlled-value suggestions:

- `visual_evidence_quality`: `clear`, `weak`, `ambiguous`, `not_visible`
- `saliency_alignment`: `aligned`, `partially_aligned`, `misaligned`, `absent`, `uncertain`
- `failure_cause_hypothesis`: `weak_visual_cue`, `small_object`, `background_confusion`, `localization_shift`, `class_confusion`, `possible_label_noise`, `cam_method_uncertain`, `other`
- `label_quality`: `ok`, `questionable`, `likely_noise`, `uncertain`
- `recommended_action`: `hard_sample_weighting`, `chipped_focused_augmentation`, `background_negative_mining`, `label_review`, `cross_method_xai_check`, `no_action`

## Decision Gate for Phase 8

Phase 8 intervention design should proceed only when manual review supports a repeated and interpretable pattern across multiple rows.

Recommended gate:

- `hard_sample_weighting` only if several `fn_low_evidence` rows are consistently judged as real hard misses rather than label issues
- `chipped_focused_augmentation` only if multiple rows repeatedly suggest weak visual cues or small-object visibility limits
- `background_negative_mining` only if false-positive rows repeatedly align with background-driven evidence patterns
- `label_review` before any intervention when label quality is often marked `questionable` or `likely_noise`
- `cross_method_xai_check` before any intervention when many rows are marked `cam_method_uncertain` or `saliency_alignment=uncertain`
- `no_action` when the manual-review results remain mixed, sparse, or not reproducible across buckets

The key rule is to move to Phase 8 only after manual review identifies a stable descriptive pattern, not because a few cases look compelling.

## Intervention Candidates

The current Phase 7 deliverables define candidate interventions only as hypotheses for later testing:

- `hard_sample_weighting`
- `chipped_focused_augmentation`
- `background_negative_mining`
- `label_review`
- `cross_method_xai_check`
- `no_action`

These are design candidates, not validated improvements.

## Limitations

- Phase 7 still depends on Phase 5 `EigenCAM` evidence only.
- Manual review is subjective and reviewer-dependent.
- Representative cases were curated in Phase 6, so the review set is useful for hypothesis formation but not for broad statistical claims.
- Some rows may reflect taxonomy ambiguity, overlap ambiguity, or label quality issues rather than pure detector failure.
- Without a filled manual review file, the current phase produces the workflow and schema but not an aggregated reviewer summary yet.

## Interpretation Boundary

- Phase 7 does not prove that the detector causally used the highlighted regions.
- Phase 7 does not prove that any intervention candidate will improve performance.
- Phase 7 does not train or evaluate any intervention.
