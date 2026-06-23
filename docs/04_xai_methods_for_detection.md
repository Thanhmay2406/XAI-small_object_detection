# 04 - XAI Methods for Detection

## Goal of this document

This document lists candidate explanation families that may later be used for evidence analysis or evidence-guided training. The purpose is to preserve methodological flexibility rather than commit early to a single explanation method.

## Candidate XAI method groups

### Gradient-based methods

These methods use gradients with respect to features, logits, objectness, or detection outputs.

Examples:

- Grad-CAM
- Grad-CAM++
- LayerCAM
- Integrated Gradients
- Input-gradient or feature-gradient saliency

Potential strengths:

- Often more target-aware than simple activation summaries.
- May better reflect which features support a particular prediction.

Potential weaknesses:

- More computationally expensive.
- Can be unstable across layers or targets.
- Harder to use inside a training loop without added overhead.

### Activation-based methods

These methods summarize activations without needing a full backward pass.

Examples:

- Channel mean activation maps
- Absolute activation mean
- EigenCAM-like projections
- Energy-based feature summaries

Potential strengths:

- Fast and easy to debug.
- Convenient for large-scale evidence analysis.
- More practical for early-phase diagnostics.

Potential weaknesses:

- May be weakly tied to a specific detection target.
- Can reflect generic feature strength instead of causal support.

### Perturbation-based methods

These methods estimate importance by masking, corrupting, or modifying regions and observing score changes.

Examples:

- Occlusion sensitivity
- Sliding-mask perturbation
- Region deletion or insertion curves
- Counterfactual masking

Potential strengths:

- Often intuitive and model-agnostic.
- Can provide stronger faithfulness evidence in some settings.

Potential weaknesses:

- Very expensive for dense detection tasks.
- Hard to scale to routine training-time use.
- Sensitive to perturbation design.

### Detection-specific methods

These methods are tailored to object detection outputs, feature heads, anchors, or queries.

Examples:

- Explanations tied to objectness and class heads
- Box-specific Grad-CAM variants
- ODAM-style detector explanations
- Query-level explanations for transformer detectors

Potential strengths:

- Better aligned with the structure of detection predictions.
- More likely to separate objectness, class evidence, and localization behavior.

Potential weaknesses:

- Less transferable across detector families.
- May increase implementation complexity early in the project.

## Why no single XAI method should be selected yet

At this stage, the repository should not commit to one explanation method for three reasons.

First, explanation maps differ in computational cost, target specificity, and stability. Choosing too early could bias both the diagnostic story and the implementation plan.

Second, the project question is not "which XAI method is best?" but "can explanation-derived evidence help small object detection?" That broader question should remain open until baseline evidence analysis clarifies what kind of signal is actually useful.

Third, some methods may be suitable for offline analysis but not for training-time regularization. A method that looks visually convincing may still be too slow or too noisy for optimization.

## Saliency map sanity-check criteria

Before any explanation method is trusted for evidence analysis, it should be checked against a basic sanity checklist.

- Spatial plausibility: salient regions should not systematically ignore the object area.
- Layer consistency: high-resolution and low-resolution maps should be interpretable after resizing, not dominated by artifacts.
- Target sensitivity: maps should change when the explained prediction or target changes.
- Model sensitivity: maps should not remain almost identical under meaningful weight changes.
- Numerical stability: normalization should avoid NaN, Inf, and degenerate all-zero outputs.
- Robustness to resizing: conclusions should not depend entirely on one interpolation choice.
- Practical cost: runtime and memory should be reasonable for the intended phase, especially if used during training.

## Working recommendation

The early project stages should keep at least two explanation families available for comparison:

- one lightweight activation-based family for scalable diagnostics,
- one more target-aware method for later validation.

This keeps the research story honest while avoiding premature lock-in.
