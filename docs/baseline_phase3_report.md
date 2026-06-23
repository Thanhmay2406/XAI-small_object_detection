# Phase 3 Baseline Report

## Goal

Establish a stable YOLO baseline on the drill-bit dataset before any XAI- or evidence-guided method is introduced.

## Scope

- Baseline detector only
- No XAI loss
- No saliency module
- No detector architecture redesign

## Baseline setup

- Dataset config: `configs/dataset/drill_bit_yolo.yaml`
- Training config: `configs/train/baseline_drill_bit_yolov8n.yaml`
- Default model: `yolov8n.pt`
- Experiment output: `experiments/baseline_drill_bit/`
- Evaluation output: `artifacts/baseline_eval/`

## Commands

### Full training

```bash
PYTHONPATH=src .venv/bin/python scripts/train_baseline.py \
  --config configs/train/baseline_drill_bit_yolov8n.yaml \
  --device cpu
```

### Smoke test

```bash
PYTHONPATH=src .venv/bin/python scripts/train_baseline.py \
  --config configs/train/baseline_drill_bit_yolov8n.yaml \
  --device cpu \
  --epochs 1 \
  --imgsz 320 \
  --batch 8 \
  --fraction 0.01
```

### Evaluation

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_baseline.py \
  --config configs/train/baseline_drill_bit_yolov8n.yaml \
  --weights experiments/baseline_drill_bit/weights/best.pt \
  --device cpu
```

## Required evaluation artifacts

- overall metrics JSON
- per-class metrics summary
- confusion matrix and PR plots from Ultralytics validation
- prediction sample images
- `Chipped` error-case CSV

## Phase 3.5 sanity helpers

- Empty-label spot-check: `scripts/spot_check_empty_labels.py`
- Error-case sampling: `scripts/sample_error_cases.py`
- Dataset sanity notes: `docs/phase3_5_dataset_sanity.md`

Recommended commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/spot_check_empty_labels.py \
  --data configs/dataset/drill_bit_yolo.yaml \
  --output artifacts/dataset_spotcheck_empty \
  --num-samples 64 \
  --seed 0
```

```bash
PYTHONPATH=src .venv/bin/python scripts/sample_error_cases.py \
  --csv artifacts/baseline_eval_smoke/chipped_error_cases.csv \
  --output artifacts/baseline_error_samples \
  --num-samples 64 \
  --seed 0
```

## Dataset framing note

This dataset is not strongly tiny-object-focused at `imgsz=640`. It is still useful for:

- baseline establishment,
- weak-evidence error analysis,
- class-specific follow-up on `Chipped`.

If later phases need stronger claims about tiny-object preservation, an additional more small-object-heavy dataset should be considered.

Smoke metrics from `baseline_drill_bit_smoke` are readiness-only artifacts and should not be used as scientific baseline results.
