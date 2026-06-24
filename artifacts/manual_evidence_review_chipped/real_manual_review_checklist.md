# Real Manual Review Checklist

Use this checklist before replacing the current demo manual review with real review content.

## Before You Start

- Open `artifacts/manual_evidence_review_chipped/manual_review_real_template.csv`
- Keep `artifacts/manual_evidence_review_chipped/manual_review_filled.csv` unchanged until the real review is ready
- Read `artifacts/manual_evidence_review_chipped/manual_review_guide.md`
- Read `docs/phase8_5_real_manual_review.md`

## Priority Order

- Review `localization_misaligned_evidence` first
- Review `near_threshold_high_evidence` second
- Then continue with the remaining buckets

## For Each Row

- Inspect `overlay_path`
- Inspect `crop_path`
- Fill `visual_evidence_quality`
- Fill `saliency_alignment`
- Fill `failure_cause_hypothesis`
- Fill `label_quality`
- Fill `recommended_action`
- Fill `reviewer_notes`
- Prefer `uncertain` or `ambiguous` over guessing

## Do Not Do

- Do not copy the synthetic smoke-demo values into the real review file
- Do not write causal claims
- Do not mark the review complete if many required fields are still blank

## Validation

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_manual_review_real.py \
  --manual-review artifacts/manual_evidence_review_chipped/manual_review_filled.csv \
  --output artifacts/manual_evidence_review_chipped/manual_review_real_validation.json
```

Validation must pass before rerunning Phase 7 summary or Phase 8 decision design.
