# 05 - Evidence-Guided Training Design

## What "evidence" means in this project

In this repository, evidence refers to the amount and quality of model support assigned to the spatial region of a target object, especially a small one. Evidence does not have to come from a single mathematical source. It is a working concept that may be instantiated in several compatible ways.

At a high level, evidence should answer:

> Is the detector preserving object-relevant support in the region where the object actually exists?

This support may be measured per feature level, per predicted instance, or per ground-truth object.

## General training objective

The general form of the training objective is:

```text
L_total = L_det + lambda * L_evidence
```

Where:

- `L_det` is the standard detection objective of the chosen baseline detector.
- `L_evidence` is an auxiliary term that encourages stronger or more stable object-relevant evidence.
- `lambda` controls how strongly evidence guidance influences training.

This definition is intentionally broad. The exact form of `L_evidence` should be derived only after baseline evidence analysis has shown what kind of degradation is measurable and relevant.

## Possible evidence sources

### XAI saliency

Saliency maps derived from activations, gradients, or detector-specific explanations can estimate how much support is placed inside the object region versus outside it.

Possible use:

- preserve in-box evidence across feature levels,
- penalize excessive evidence drop,
- reduce background-dominant explanations.

### Bbox-derived soft masks

Ground-truth boxes can be converted into hard or soft spatial masks that define where object evidence should be concentrated.

Possible use:

- compare saliency concentration inside vs outside the mask,
- support expanded or uncertainty-aware object regions,
- reduce brittleness for very tiny boxes.

### Feature energy

Feature magnitude or energy can be used as a proxy for support even without an explicit saliency method.

Possible use:

- measure whether object regions retain activation energy,
- build lightweight evidence terms for faster experimentation,
- provide a fallback when explanation methods are too expensive.

### Objectness response

The detector's own objectness or confidence-related outputs may serve as a more prediction-aware evidence source.

Possible use:

- tie evidence preservation to regions expected to contain objects,
- distinguish object support from generic activation intensity.

### Teacher explanation

A teacher model or a stronger explanation pipeline can provide pseudo-target explanations to guide a student detector.

Possible use:

- align student evidence with teacher evidence,
- distill object-centered support across levels.

This option is promising but should be treated as a later-stage extension, not a starting assumption.

## Design principles

- Evidence guidance must not break baseline training when `lambda = 0`.
- The auxiliary term should focus on measurable failure modes, not abstract heatmap aesthetics.
- The design should remain model-agnostic enough to work with more than one detector family later.
- Evidence loss should be compatible with fair ablations over level, method, and object size group.

## Risks and failure cases

### Misleading explanations

If the chosen evidence source is not faithful to the detector's decision process, optimizing it may reward visually appealing but functionally irrelevant behavior.

### Over-regularization

A strong auxiliary term may suppress useful flexibility in the detector, hurting AP, precision, or convergence.

### Size-group imbalance

A loss designed only for small objects may help AP_S while harming medium or large objects. That tradeoff must be measured rather than assumed away.

### Background leakage through masks

Bounding boxes are coarse object descriptions. A naive in-box objective may encourage attention to background pixels inside the box rather than true object structure.

### Compute overhead

Some evidence sources are too expensive for dense training loops. The project should consider lightweight evidence definitions before committing to high-cost explanation methods.

### Architecture dependence

An evidence term that works only for one head design or one feature naming scheme may not generalize. This is another reason to keep the design abstract at the documentation stage.

## What this document does not decide yet

This document does not fix:

- one exact saliency method,
- one exact evidence formula,
- one exact baseline detector architecture,
- one exact set of feature levels.

Those choices should be made only after the dataset and baseline analysis stages provide enough evidence to justify them.
