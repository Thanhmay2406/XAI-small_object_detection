# Phase 3.5 Dataset Sanity

## Goal

Sanity-check the drill-bit dataset and baseline-readiness artifacts before any evidence-guided or XAI-guided method is introduced.

## Scope

- Empty-label image spot-check
- Baseline error-case sampling
- Command/path verification for full baseline training and evaluation

## Commands

### Empty-label spot-check

```bash
PYTHONPATH=src .venv/bin/python scripts/spot_check_empty_labels.py \
  --data configs/dataset/drill_bit_yolo.yaml \
  --output artifacts/dataset_spotcheck_empty \
  --num-samples 64 \
  --seed 0
```

### Error-case sampling

```bash
PYTHONPATH=src .venv/bin/python scripts/sample_error_cases.py \
  --csv artifacts/baseline_eval_smoke/chipped_error_cases.csv \
  --output artifacts/baseline_error_samples \
  --num-samples 64 \
  --seed 0
```

## Manual review checklist

### Empty-label images

- Confirm that sampled empty-label images are genuine negative examples.
- Look for missed defect regions, especially subtle `Chipped` cases.
- Check whether empty-label images cluster by lighting condition, crop pattern, or split.

### Chipped error samples

- Review whether false negatives are visually weak-evidence cases.
- Look for small, low-contrast, or edge-fragment defects that a plain YOLO baseline may miss.
- Track whether a subset of `Chipped` errors looks systematically under-labeled or ambiguous.

## Baseline command verification

- Full training target directory: `experiments/baseline_drill_bit/`
- Full evaluation checkpoint: `experiments/baseline_drill_bit/weights/best.pt`
- Evaluation artifact directory: `artifacts/baseline_eval/`

## Reminder

Smoke metrics are only readiness checks. They must not be reported as scientific results.
