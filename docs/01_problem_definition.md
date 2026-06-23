# 01 - Problem Definition

## Problem scope

This project studies supervised object detection in scenes where relevant targets occupy only a small fraction of the image area. In practical terms, small objects may span only a few pixels to a few tens of pixels after resizing to the detector input resolution.

The central problem is not simply low accuracy on a difficult subset. It is that small objects often provide weak and unstable evidence to the detector, making them disproportionately vulnerable to being ignored, mislocalized, or confused with background structure.

## Why small objects are difficult for CNN and YOLO-style detectors

CNN and YOLO-style detectors rely on spatial downsampling, feature aggregation, and prediction over grids or feature levels. These mechanisms are effective for efficiency and semantics, but they can hurt small objects in several ways.

### Feature degradation through downsampling

Small objects can lose distinguishable structure after repeated stride reductions. A target that is already tiny in the input image may map to very few cells in higher-stride features, reducing the model's ability to preserve shape, boundary, and localization cues.

### Weak object signal

The visual signal of a small object is often dominated by low pixel count, weak texture, compression noise, motion blur, or atmospheric effects. As a result, the object contribution may be weaker than surrounding context even before deep feature extraction begins.

### Background bias

When object evidence is weak, the detector may rely on surrounding context, repetitive textures, or high-contrast background patterns. This can create apparently confident predictions that are not strongly grounded in the true object region.

### Localization sensitivity

Bounding-box localization is especially fragile for small objects. A small shift in center or width/height can cause a large IoU drop. Because evaluation thresholds operate on overlap, minor coordinate errors can change a true positive into a localization error or false negative.

### Multi-scale mismatch

Feature pyramids help by providing higher-resolution levels, but the project assumes that multi-scale design alone does not guarantee evidence preservation. A detector may still have the right feature levels available while failing to maintain object-relevant evidence consistently across them.

## Working interpretation for this repository

Within this repository, the small object detection problem is defined as:

> A detection setting in which object-relevant evidence is weak, easily degraded by feature processing, and highly sensitive to background distraction and localization error.

This definition is intentionally broader than a single area threshold. A configurable size threshold will still be required for experiments, but the scientific framing goes beyond a dataset-specific cutoff.

## Project-specific technical concerns

The documentation and later experiments should explicitly account for:

- Feature degradation across backbone and FPN levels.
- Reduced signal-to-background contrast for small instances.
- Over-reliance on contextual cues instead of object-centered evidence.
- Sensitivity of IoU-based evaluation to small localization shifts.
- The possibility that a detector predicts correctly while still relying on the wrong visual evidence.

## Implication for the research direction

Because the failure mechanism may involve evidence loss rather than only insufficient capacity, this project investigates whether explanation-derived signals can serve as an auxiliary training objective. The goal is not to replace standard detection loss, but to augment it with a signal that encourages the detector to preserve small-object evidence more faithfully.
