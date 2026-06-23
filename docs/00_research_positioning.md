# 00 - Research Positioning

## Research motivation

Small object detection remains difficult even when modern detectors already use multi-scale features, stronger backbones, or improved neck designs. The persistent issue is not only that small objects are hard to classify, but that their supporting evidence can be weak, spatially fragile, and easily diluted as features move through downsampling and fusion stages.

This repository is motivated by a simple research idea:

> If we can measure whether a detector is preserving object-relevant evidence for small objects, we may be able to use that signal to improve training.

The project therefore sits at the intersection of three areas:

- Small object detection, where resolution loss and localization fragility are dominant challenges.
- Explainable AI, where saliency or attribution can expose which regions and features influence model decisions.
- Training-time model improvement, where explanation-derived signals may act as regularizers rather than post-hoc visual aids.

## Positioning of the project

This work is not framed as a new detector architecture proposal. The immediate goal is not to invent another backbone, neck, or attention block. Instead, the project asks whether XAI-derived evidence can help diagnose and regularize an existing detector during training.

The intended progression is:

1. Establish a reliable baseline detector.
2. Analyze how small-object evidence behaves across feature levels.
3. Test whether evidence-aware training objectives reduce harmful evidence degradation.
4. Evaluate whether that reduction translates into better small-object detection metrics.

In that sense, the project is closer to a training and analysis framework than to an architecture paper.

## Scientific contribution space

The contribution space of this repository is expected to include:

- A practical definition of small-object evidence that can be computed from model features or explanations.
- A diagnostic protocol for measuring evidence preservation or evidence drop across levels such as P2, P3, and P4.
- A training design in which detection loss is augmented by an evidence-guided term.
- An experimental story linking explanation behavior to detection outcomes such as AP_S, AR_S, and false negatives.

## Why not commit to a new architecture yet

Architecture changes can improve small object detection, but they make it harder to answer the scientific question cleanly. If backbone changes, neck changes, and evidence-based regularization are introduced together, we cannot tell which factor caused any observed gain.

For that reason, this repository deliberately starts from the following assumptions:

- The baseline detector should remain interpretable and independently runnable.
- Evidence analysis must exist before evidence-guided training is added.
- Training-time evidence signals should first be tested on a stable detector family before broader architectural changes are considered.

## Research stance

The project adopts the following stance:

- XAI is not used only for pretty heatmaps.
- Explanation quality alone is not the target outcome.
- The main goal is to test whether explanation-derived evidence can support learning for small object detection.
- The final claim should be based on measurable improvements and controlled ablations, not visual intuition alone.
