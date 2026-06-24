# Phase 5 XAI Evidence Extraction

## Scope

Phase 5 adds a post-hoc XAI evidence pipeline on top of the locked baseline checkpoint and curated `Chipped` cases.

This phase does not:

- train a new model,
- add XAI loss,
- modify the YOLO architecture,
- claim that XAI improves detection performance.

## Inputs

- Dataset config: `configs/dataset/drill_bit_yolo.yaml`
- Locked baseline checkpoint: `experiments/baseline_drill_bit/weights/best.pt`
- Focus error cases: `artifacts/baseline_error_analysis/focus_class_error_cases.csv`
- Exported prediction rows: `artifacts/baseline_eval/prediction_rows.csv`

## What was implemented

Source modules:

- `src/xai_evidence_sod/xai/case_selection.py`
- `src/xai_evidence_sod/xai/cam.py`
- `src/xai_evidence_sod/xai/evidence_metrics.py`
- `src/xai_evidence_sod/xai/evidence_pipeline.py`

CLI:

- `scripts/extract_xai_evidence.py`

Current method support:

- Implemented: `eigencam`
- Scaffolded for later: `gradcam`, `gradcam++`

## Curated-case design

The Phase 5 selector builds a balanced `Chipped` subset from:

- `false_negative`
- `false_positive`
- `localization_error`
- `true_positive_proxy`

It also preserves a `near_threshold_overlap` flag when a case has sufficiently high overlap to be review-worthy.

`true_positive_proxy` is reconstructed from exported predictions plus test-set labels. It is suitable for comparative review, but it is still a proxy rather than a native validator export.

## Evidence metrics

For each extracted CAM, the pipeline computes:

- `energy_in_gt_box`
- `energy_in_pred_box`
- `peak_inside_gt_box`
- `saliency_concentration`

Definitions:

- `energy_in_gt_box`: fraction of normalized CAM energy inside the GT box.
- `energy_in_pred_box`: fraction of normalized CAM energy inside the predicted box when available.
- `peak_inside_gt_box`: whether the highest-energy pixel lies inside the GT box.
- `saliency_concentration`: fraction of total CAM energy contained in the top 5 percent of CAM pixels.

These are descriptive evidence summaries only. They should not be interpreted as causal proof.

## Commands used

Sanity:

```bash
.venv/bin/python -m compileall src scripts
```

Smoke extraction:

```bash
PYTHONPATH=src .venv/bin/python scripts/extract_xai_evidence.py \
  --weights experiments/baseline_drill_bit/weights/best.pt \
  --data configs/dataset/drill_bit_yolo.yaml \
  --cases artifacts/baseline_error_analysis/focus_class_error_cases.csv \
  --output artifacts/xai_evidence_chipped \
  --focus-class Chipped \
  --methods eigencam \
  --max-cases 16 \
  --seed 0
```

Full Phase 5 extraction:

```bash
PYTHONPATH=src .venv/bin/python scripts/extract_xai_evidence.py \
  --weights experiments/baseline_drill_bit/weights/best.pt \
  --data configs/dataset/drill_bit_yolo.yaml \
  --cases artifacts/baseline_error_analysis/focus_class_error_cases.csv \
  --output artifacts/xai_evidence_chipped \
  --focus-class Chipped \
  --methods eigencam \
  --max-cases 64 \
  --seed 0
```

## Full extraction snapshot

From `artifacts/xai_evidence_chipped/evidence_summary.json`:

- Method: `eigencam`
- Target layer index: `18`
- Selected cases: `64`
- Group balance:
- `false_negative`: `16`
- `false_positive`: `16`
- `localization_error`: `16`
- `true_positive_proxy`: `16`
- `near_threshold_overlap` flagged cases: `24`

Aggregate evidence stats:

- average `saliency_concentration`: `0.3340`
- average `energy_in_gt_box`: `0.0802`
- average `energy_in_pred_box`: `0.0794`

Observed note:

- `peak_inside_gt_box` is true only for a minority of rows in the current `64`-case extraction, which supports the decision to keep this phase descriptive and exploratory.

## Output artifacts

- `artifacts/xai_evidence_chipped/evidence_cases.csv`
- `artifacts/xai_evidence_chipped/evidence_summary.json`
- `artifacts/xai_evidence_chipped/overlays/`
- `artifacts/xai_evidence_chipped/crops/`
- `artifacts/xai_evidence_chipped/maps/`
- `artifacts/xai_evidence_chipped/contact_sheets/`
- `artifacts/xai_evidence_chipped/README.md`

## Interpretation boundaries

- Phase 5 shows where a stable post-hoc XAI pipeline can run on the locked baseline.
- Phase 5 does not prove that EigenCAM explanations are faithful in a strict causal sense.
- Phase 5 does not prove that adding explanation-derived loss would improve the detector.
- The dataset should still be framed as weak-evidence / hard-class detection, not as a strongly tiny-object benchmark.

## Recommended Phase 6

Proceed to a focused evidence-review phase that:

- compares `true_positive_proxy` vs `false_negative` vs `localization_error` rows,
- manually reviews the highest-value `near_threshold_overlap` cases,
- decides whether `Grad-CAM` and `Grad-CAM++` should be added for contrastive inspection,
- prepares a narrower set of evidence hypotheses before any training-time evidence regularization is attempted.
