# Phase 4 Error Analysis And Evidence Plan

## Purpose

Phase 4 locks the completed full baseline as the official comparison point, analyzes where it fails, and prepares artifact structure for later XAI/evidence inspection.

This phase does not:

- add XAI loss,
- train a new model,
- change the YOLO architecture,
- claim that XAI improves the detector.

## Official baseline summary

Primary baseline artifacts:

- `artifacts/baseline_eval/metrics_overall.json`
- `artifacts/baseline_eval/prediction_rows.csv`
- `artifacts/baseline_eval/chipped_error_cases.csv`
- `artifacts/baseline_eval/val/`

Official full baseline metrics:

- mAP50-95: `0.3686`
- mAP50: `0.6965`
- mAP75: `0.3314`

Per-class AP50-95:

- `Broken`: `0.5663`
- `Chipped`: `0.2685`
- `Scratched`: `0.2874`
- `Severe_Rust`: `0.2742`
- `Tip_Wear`: `0.4464`

Interpretation:

- The baseline is usable as a comparison anchor.
- `Chipped` is the weakest and most research-relevant class.
- This remains a weak-evidence / hard-class framing, not a strong tiny-object benchmark framing.

## Error taxonomy used in Phase 4

The new analysis script uses post-hoc box matching and heuristic tags derived from GT labels plus exported prediction rows.

Primary error types:

- `false_negative`
- `false_positive`
- `localization_error`

Secondary heuristic tags:

- `class_confusion`
- `low_confidence_detection`
- `small_or_weak_evidence_case`
- `ambiguous_or_possible_label_noise`

Important note:

- `localization_error`, `class_confusion`, and `ambiguous_or_possible_label_noise` are heuristic review categories, not causal conclusions.
- The current artifacts do not contain saliency maps or feature-level evidence, so Phase 4 only prepares the next step.

## Chipped-focused analysis

From `artifacts/baseline_error_analysis/focus_class_error_summary.json`:

- GT count: `197`
- TP proxy: `118`
- FN count: `79`
- FP count: `78`
- AP50-95: `0.2685`
- Precision proxy: `0.6020`
- Recall proxy: `0.5990`

Focused findings:

- `79` ground-truth `Chipped` objects were missed, matching the earlier exported `chipped_error_cases.csv`.
- `92` `Chipped` error rows had zero overlap with the best retained prediction, so many misses are outright failures rather than small localization slips.
- `26` `Chipped` error rows were near the IoU threshold and are the strongest candidates for future evidence-map analysis.
- `49` `Chipped` error rows were tagged as `small_or_weak_evidence_case`.
- `46` `Chipped` error rows involved `low_confidence_detection`.
- `11` `Chipped` error rows showed heuristic `class_confusion`.

Research interpretation:

- `Chipped` is still the best target class for evidence-guided follow-up because it combines low AP with a meaningful mix of outright misses and near-hit borderline cases.
- The near-threshold `Chipped` subset is especially valuable for later saliency/evidence inspection because the detector is responding, but not robustly enough.
- Some cross-class overlaps suggest that part of the difficulty may come from ambiguous visual boundaries or annotation ambiguity, not only lack of evidence.

## Artifacts produced in Phase 4

Error analysis:

- `artifacts/baseline_error_analysis/error_summary.json`
- `artifacts/baseline_error_analysis/per_class_error_summary.csv`
- `artifacts/baseline_error_analysis/focus_class_error_summary.json`
- `artifacts/baseline_error_analysis/focus_class_error_cases.csv`
- `artifacts/baseline_error_analysis/size_bin_error_summary.csv`
- `artifacts/baseline_error_analysis/confidence_error_summary.csv`
- `artifacts/baseline_error_analysis/README.md`

Manual review gallery:

- `artifacts/baseline_error_gallery/sampled_errors.csv`
- `artifacts/baseline_error_gallery/previews/`
- `artifacts/baseline_error_gallery/crops/`
- `artifacts/baseline_error_gallery/error_gallery_contact_sheet.jpg`
- `artifacts/baseline_error_gallery/README.md`

## Limitations of current baseline artifacts

- The analysis uses exported prediction rows plus post-hoc IoU matching, not internal validator assignments.
- Current artifacts do not include feature activations, explanation maps, or per-layer response traces.
- `class_confusion` and `ambiguous_or_possible_label_noise` are only heuristic tags from overlap patterns.
- The current dataset should not be presented as a strongly tiny-object-focused benchmark.

## Phase 5 hook design proposal

Before generating real saliency maps, the repo should define which prediction paths and feature locations will be inspected first.

Recommended starting hooks:

- final detection head outputs associated with `Chipped`
- one mid/high-resolution feature level that still preserves local defect cues
- one lower-resolution semantic feature level for context comparison

Recommended first review targets:

- `Chipped` false negatives with non-zero overlap
- `Chipped` false positives with moderate confidence
- `Chipped` cases tagged with `class_confusion`
- a small control set of correct `Chipped` true positives

Recommended XAI methods for Phase 5:

- Grad-CAM
- Grad-CAM++
- EigenCAM
- LayerCAM if spatial detail is needed
- D-RISE or another perturbation-based method for selective verification

## Phase 5 recommendation

Proceed to a focused evidence-analysis phase that:

- keeps the same trained baseline weights,
- extracts explanation maps on curated `Chipped` cases,
- compares correct detections, false negatives, false positives, and near-threshold cases,
- avoids any training-time evidence loss until the evidence patterns are better understood.
