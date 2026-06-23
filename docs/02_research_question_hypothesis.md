# 02 - Research Question and Hypotheses

## Main research question

> Can XAI-derived or evidence-derived training signals help a detector preserve small-object evidence across feature levels, and does that preservation improve small-object detection performance?

This question has two linked parts:

- A diagnostic part: whether evidence degradation is measurable in a baseline detector.
- An intervention part: whether evidence-guided training can reduce that degradation and improve outcomes.

## Testable hypotheses

### H1 - Baseline evidence degradation hypothesis

For small objects, evidence measured at higher-resolution feature levels will tend to be stronger than evidence measured at deeper, lower-resolution levels.

Expected observation:

- Mean evidence at P2 is higher than at P3.
- Mean evidence at P3 is higher than at P4.
- The degradation trend is stronger for small objects than for medium or large objects.

### H2 - Failure association hypothesis

Small objects associated with false negatives or localization errors will show weaker in-box evidence or larger cross-level evidence drop than small objects that are correctly detected.

Expected observation:

- Missed small objects have lower evidence concentration in object regions.
- Missed or poorly localized objects show larger evidence drop between adjacent feature levels.

### H3 - Performance hypothesis

Adding an evidence-guided term to the training objective will improve small-object performance relative to the baseline under matched training settings.

Expected observation:

- AP_S increases.
- AR_S increases.
- Small-object false negatives decrease.

### H4 - Convergence and stability hypothesis

Moderate evidence-guided regularization can improve or stabilize training behavior by providing an auxiliary signal when object evidence is weak.

Expected observation:

- Validation AP_S becomes less erratic across epochs or seeds.
- Training avoids obvious collapse in localization or recall.
- Evidence-related statistics become more consistent during training.

### H5 - Over-regularization hypothesis

If the evidence term is too strong or poorly defined, it can bias the detector toward misleading explanations or reduce overall detection quality.

Expected observation:

- Large lambda values may hurt AP or precision.
- Some XAI definitions may create unstable optimization or reward visually plausible but unhelpful saliency patterns.

## Secondary research questions

- Which feature levels are most informative for evidence preservation in small object detection?
- Do different explanation families lead to similar evidence trends, or are conclusions highly method-dependent?
- Does evidence-guided training help only tiny and small objects, or does it affect medium and large objects as well?
- Is the method robust across random seeds and small changes in setup?

## What would count as support

The research direction is supported if the project can show all of the following:

- Evidence degradation is measurable in the baseline.
- The measured degradation relates to meaningful failure modes.
- Evidence-guided training reduces at least part of that degradation.
- Performance gains survive fair baseline comparison and limited ablation.

## What would count as a negative result

The project should also be prepared for a valid negative outcome:

- Evidence degradation may be measurable but not actionable.
- Evidence-guided loss may change explanations without improving detection.
- Improvement may depend too strongly on one XAI method or one seed.

Such outcomes would still be scientifically useful because they clarify the boundary between diagnostic explanation and trainable explanation.
