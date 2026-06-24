# Phase 6 Evidence Review and Failure-Mode Comparison

## Goal

Phase 6 reviews the Phase 5 EigenCAM evidence artifacts for the `Chipped` class without retraining the detector, changing the YOLO architecture, or adding any XAI-derived loss.

This phase is descriptive only. It compares post-hoc evidence patterns across case groups and prepares a manual-review shortlist for later research decisions.

## Inputs

- `artifacts/xai_evidence_chipped/evidence_cases.csv`
- `artifacts/xai_evidence_chipped/evidence_summary.json`
- `artifacts/xai_evidence_chipped/overlays/`
- `artifacts/xai_evidence_chipped/contact_sheets/`

## Outputs

- `artifacts/xai_evidence_review_chipped/evidence_group_summary.csv`
- `artifacts/xai_evidence_review_chipped/evidence_group_summary.json`
- `artifacts/xai_evidence_review_chipped/representative_cases.csv`
- `artifacts/xai_evidence_review_chipped/review_notes_template.csv`
- `artifacts/xai_evidence_review_chipped/README.md`
- `artifacts/xai_evidence_review_chipped/evidence_group_means.png`
- `artifacts/xai_evidence_review_chipped/peak_inside_gt_rate.png`

## Implementation

Source additions:

- `src/xai_evidence_sod/xai/evidence_review.py`
- `scripts/review_xai_evidence.py`

CLI used:

```bash
.venv/bin/python -m compileall src scripts
PYTHONPATH=src .venv/bin/python scripts/review_xai_evidence.py \
  --evidence-csv artifacts/xai_evidence_chipped/evidence_cases.csv \
  --output artifacts/xai_evidence_review_chipped \
  --focus-class Chipped \
  --top-k 8
```

Behavior notes:

- The review script keeps paths relative in the exported review artifacts.
- Missing metric columns trigger clear warnings and produce null summary fields instead of crashing.
- Optional plots are only exported when `matplotlib` is available.

## Quantitative Findings

Reviewed rows:

- `64` total `Chipped` evidence rows
- `16` `true_positive_proxy`
- `16` `false_negative`
- `16` `false_positive`
- `16` `localization_error`
- `24` `near_threshold_overlap` rows flagged by `is_near_threshold=True`

### Group summary snapshot

| group | count | mean gt energy | median gt energy | mean pred energy | peak-inside-gt rate | mean saliency concentration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true_positive_proxy | 16 | 0.0393 | 0.0033 | 0.0408 | 0.1250 | 0.2400 |
| false_negative | 16 | 0.0072 | 0.0054 | 0.0669 | 0.0000 | 0.2891 |
| false_positive | 16 | 0.2159 | 0.1904 | 0.1029 | 0.5000 on 6 observed rows | 0.4456 |
| localization_error | 16 | 0.1432 | 0.0461 | 0.0967 | 0.5000 | 0.3614 |
| near_threshold_overlap | 24 | 0.0821 | 0.0178 | 0.0691 | 0.2917 | 0.2712 |

### Observed patterns

1. `false_negative` rows have the weakest GT-box evidence in this Phase 5 sample.
   - mean `energy_in_gt_box = 0.0072`
   - median `energy_in_gt_box = 0.0054`
   - `peak_inside_gt_box_rate = 0.0`

2. `false_negative` rows still show non-trivial predicted-box energy when a predicted box exists.
   - mean `energy_in_pred_box = 0.0669`
   - the strongest misaligned examples include `error_0028`, `error_0025`, and `error_0020`
   - this supports a descriptive reading of "evidence present but not aligned with the GT box" for a subset of misses

3. `localization_error` rows sit between clean misses and stronger-evidence rows.
   - mean `energy_in_gt_box = 0.1432`
   - mean `energy_in_pred_box = 0.0967`
   - `peak_inside_gt_box_rate = 0.5`
   - near-threshold localization rows such as `error_0107` and `error_0122` are especially review-worthy

4. `near_threshold_overlap` is mixed rather than uniformly high-evidence or low-evidence.
   - mean `energy_in_gt_box = 0.0821`
   - median `energy_in_gt_box = 0.0178`
   - `peak_inside_gt_box_rate = 0.2917`
   - the high-evidence end includes `error_0107` and `tp_0106`
   - the low-evidence end includes `tp_0065`, `tp_0061`, and `tp_0097`

5. `true_positive_proxy` rows are heterogeneous.
   - one outlier, `tp_0106`, has very strong GT and predicted-box evidence
   - the group median GT-box energy is only `0.0033`, much lower than the group mean `0.0393`
   - this suggests the current proxy sample should be reviewed qualitatively before being treated as a stable "good evidence" reference set

6. `false_positive` rows show high concentration and, in this export, often non-zero GT-box energy.
   - mean `energy_in_gt_box = 0.2159`
   - mean `saliency_concentration = 0.4456`
   - this does not imply correct detector reasoning; it more likely reflects overlap structure, proxy taxonomy limits, or evidence focusing on visually salient but non-decisive regions

## Representative Review Cases

The Phase 6 script exports `8` rows per review bucket:

- `tp_high_evidence`
- `fn_low_evidence`
- `fn_misaligned_evidence`
- `fp_background_like_evidence`
- `localization_misaligned_evidence`
- `near_threshold_high_evidence`
- `near_threshold_low_evidence`

Top examples from the current run:

- `tp_high_evidence`: `tp_0106`, `tp_0049`, `tp_0053`
- `fn_low_evidence`: `error_0041`, `error_0013`, `error_0035`
- `fn_misaligned_evidence`: `error_0028`, `error_0025`, `error_0020`
- `fp_background_like_evidence`: `error_0051`, `error_0151`, `error_0132`
- `localization_misaligned_evidence`: `error_0107`, `error_0112`, `error_0122`
- `near_threshold_high_evidence`: `error_0107`, `tp_0106`, `error_0122`
- `near_threshold_low_evidence`: `tp_0065`, `tp_0061`, `tp_0097`

`review_notes_template.csv` is provided so manual observations can be added without editing the selection logic.

## Limitations

- The review uses Phase 5 `EigenCAM` outputs only. `Grad-CAM` and `Grad-CAM++` remain out of scope for this phase.
- Evidence metrics are post-hoc descriptive summaries, not causal proof of what the detector used.
- `true_positive_proxy` remains a proxy reconstruction rather than a native validator match export.
- `false_positive` and `localization_error` interpretations depend on the available GT/pred box pair structure in the Phase 5 CSV, so high GT-box energy in those groups should not be over-read.
- The current sample size is intentionally small and curated (`64` rows), so patterns should guide review, not claim generality.

## Decision Gate for Phase 7

Proceed to Phase 7 only if the manual review of `representative_cases.csv` supports at least one stable descriptive hypothesis such as:

- false negatives frequently show low GT-box evidence,
- some misses show stronger evidence in nearby or predicted regions than in GT regions,
- near-threshold rows separate into visually meaningful high-evidence and low-evidence subsets,
- localization errors repeatedly show displaced evidence that is reviewable across multiple cases.

Do not proceed to any training-time evidence loss solely from these Phase 6 summaries. The next step should first tighten manual annotation of the representative cases and decide whether an additional contrastive post-hoc method is needed for inspection only.
