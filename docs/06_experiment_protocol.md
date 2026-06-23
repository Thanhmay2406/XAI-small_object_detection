# 06 - Experiment Protocol

## Experiment principles

All experiments in this repository should serve the research question rather than chase isolated metric gains. The protocol should preserve a clear causal story from baseline behavior to evidence analysis to evidence-guided intervention.

Core principles:

- Measure before intervening.
- Change one major factor at a time.
- Keep baseline training independently valid.
- Record every run with config, seed, outputs, and notes.
- Avoid claims based on visualization alone.

## Baseline comparisons

The minimum comparison structure should include:

- A baseline detector trained with standard detection loss only.
- A matched variant with evidence-guided training added.
- Optional control variants that help separate evidence effects from generic auxiliary-loss effects.

Fair comparison requires that the following remain matched unless they are the subject of an ablation:

- dataset split,
- input resolution,
- optimizer and scheduler,
- augmentation policy,
- epoch budget,
- seed policy,
- checkpoint selection rule,
- evaluation settings.

The project should not attribute gains to evidence guidance if multiple unrelated training changes were introduced simultaneously.

## Evaluation metrics

Primary metrics:

- `AP_S`
- `AR_S`
- small-object false negatives

Secondary detection metrics:

- `AP`
- `AP50`
- `AP75`
- `Precision`
- `Recall`
- size-specific metrics for medium and large objects when available

Evidence-oriented metrics:

- evidence concentration inside object region,
- evidence outside object region,
- cross-level evidence drop,
- object-centered saliency hit or alignment statistics.

Training behavior metrics:

- loss curves,
- validation stability across epochs,
- instability symptoms such as NaN, divergence, or highly erratic evidence statistics.

Efficiency metrics:

- training time per epoch,
- additional memory cost,
- extra cost introduced by the evidence branch.

## Ablation dimensions

The core ablation dimensions for this project are:

- `lambda`: strength of the evidence term
- feature level: for example P2 only, P2 plus P3, or P2 plus P3 plus P4
- XAI method: activation-based, gradient-based, or detector-specific candidates
- object size group: tiny, small, or all objects
- seed: at least enough repetition to distinguish meaningful gains from variance

These ablations should be added gradually. The repository should not launch a wide ablation grid before the baseline and evidence analysis stages are validated.

## Reproducibility requirements

Each experiment should log:

- run identifier,
- exact config used,
- seed,
- code state or commit identifier if available,
- generated artifacts,
- short human-readable summary.

Outputs should be saved under repository-controlled directories such as:

- `experiments/`
- `artifacts/`
- `paper/`

## Evidence analysis before evidence loss

The protocol explicitly requires a diagnostic checkpoint before evidence-guided training is introduced. A baseline evidence analysis should answer:

- whether small-object evidence degradation is measurable,
- which feature levels matter most,
- whether detector failure modes correlate with evidence statistics.

If that analysis is weak or inconclusive, the evidence-loss design must remain tentative.

## Fair-comparison checklist

Before accepting any comparison as valid, check:

- Was the baseline rerun under the same setup?
- Were seeds handled consistently?
- Were output metrics computed the same way?
- Was the same data split used?
- Was early stopping or checkpoint selection comparable?
- Did the evidence branch add hidden data filtering or sample selection differences?

## Success criteria for the research direction

The project is on a promising path if later experiments can show:

- baseline evidence degradation exists and is measurable,
- evidence guidance reduces harmful degradation,
- small-object metrics improve under fair comparison,
- gains are not explained only by one seed or one visualization choice.
