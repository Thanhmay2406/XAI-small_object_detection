# 03 - Small Object Detection Failure Modes

## Purpose

This document defines the main failure modes that later analysis should examine before proposing evidence-guided training as a remedy.

## Failure-mode table

| Failure mode | Observable symptoms | Possible measurements | Possible XAI relevance |
|---|---|---|---|
| False negatives | Ground-truth small objects are not matched by any valid prediction; recall drops on crowded or low-contrast scenes | `FN_small`, recall by size bucket, miss rate by class or scene type | Saliency may reveal absent or weak object-centered evidence even when context is visually strong |
| Background bias | Predictions respond to nearby texture, clutter, shadows, or high-contrast context instead of the object | In-box vs out-of-box saliency ratio, false positives near object neighborhoods, context-sensitive confidence changes | Explanations can show whether attention is centered on background structure rather than the object |
| Weak saliency | The object region receives little explanation mass relative to the full image or surrounding area | Saliency concentration inside bbox, peak-in-bbox hit rate, evidence ratio per level | Indicates that the detector may not be grounding predictions in the object region |
| Localization error | Detector predicts the right class near the object but box alignment is poor, causing low IoU | IoU distribution for matched predictions, center error, width-height error, AP75 gap vs AP50 | XAI may show diffuse or off-center evidence that does not align with object boundaries |
| Scale degradation | Evidence is present at a high-resolution level but becomes weak or diffuse at deeper levels | Evidence at P2/P3/P4, cross-level evidence drop, per-level concentration change | Directly motivates the idea of evidence preservation across feature levels |
| Context over-reliance | The model predicts correctly only when surrounding context remains intact and fails when local context changes | Occlusion sensitivity around bbox, crop-vs-full-image consistency, performance under background perturbation | Explanations may highlight surrounding regions more strongly than the target itself |

## Interpretation notes

These failure modes are related but not identical. For example, a false negative may be caused by weak evidence, scale degradation, or context mismatch. Similarly, a correctly detected object can still exhibit background bias if the model uses the wrong cues.

The project should therefore avoid treating one explanation map as a complete diagnosis. Instead, later analysis should connect saliency and evidence measurements to concrete detector outcomes such as:

- true positive versus false negative,
- accurate localization versus poor localization,
- high-confidence versus low-confidence prediction,
- stable versus unstable behavior across feature levels.

## Why this matters for the repository

The evidence-guided training idea is only justified if at least some of these failure modes are observable and if evidence-based measurements add information beyond standard detection metrics. This document serves as a checklist for that claim.
