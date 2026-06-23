# Experiment Log

This log records milestone-level project activity so research decisions remain reproducible.

## 2026-06-23

### Phase 0 - Repository initialization

- Standardized the initial research repository skeleton.
- Added packaging metadata, minimal dependencies, config placeholders, and source package layout.
- Reserved directories for experiments, artifacts, tests, notebooks, and paper drafting.

### Phase 1 - Research docs

- Scope: documentation-only phase.
- No training code, XAI implementation, or model architecture changes were introduced.
- Added a new research-document set to define project positioning, problem framing, hypotheses, failure modes, XAI candidates, evidence-guided training design, and experiment protocol.

Files created:

- `docs/00_research_positioning.md`
- `docs/01_problem_definition.md`
- `docs/02_research_question_hypothesis.md`
- `docs/03_small_object_detection_failure_modes.md`
- `docs/04_xai_methods_for_detection.md`
- `docs/05_evidence_guided_training_design.md`
- `docs/06_experiment_protocol.md`

Files modified:

- `docs/experiment_log.md`

Notes:

- The documentation intentionally avoids committing early to a single detector architecture or a single XAI method.
- The research direction remains focused on using explanation-derived or evidence-derived signals to support small object detection training after baseline analysis is complete.

### Phase 2 - Dataset inspection

- Goal: implement a reproducible YOLO-format dataset inspection pipeline for split validation, bbox statistics, and research-oriented reporting.

Files created:

- `src/xai_evidence_sod/data/yolo_parser.py`
- `src/xai_evidence_sod/data/bbox_stats.py`
- `src/xai_evidence_sod/data/validation.py`
- `src/xai_evidence_sod/data/dataset_report.py`

Files modified:

- `src/xai_evidence_sod/data/__init__.py`
- `scripts/inspect_dataset.py`
- `docs/experiment_log.md`

Commands to run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/inspect_dataset.py --data configs/dataset/example_yolo.yaml --output artifacts/dataset_inspection`

Expected outputs:

- `artifacts/dataset_inspection/dataset_summary.json`
- `artifacts/dataset_inspection/bbox_size_stats.csv`
- `artifacts/dataset_inspection/class_distribution.csv`
- `artifacts/dataset_inspection/split_summary.csv`
- `artifacts/dataset_inspection/invalid_labels.csv`
- `artifacts/dataset_inspection/inspection_report.md`

Checks performed:

- Source and script syntax compiled after implementation.
- Dataset inspection command should be tested against the configured YAML and, if needed, a real dataset YAML in the repository.

Risks / open questions:

- `configs/dataset/example_yolo.yaml` may remain a placeholder if it does not point to a real dataset root.
- Missing label files can indicate either negative images or annotation gaps, so the report must be interpreted with dataset context.
- Class imbalance and size imbalance should be reviewed before baseline training begins.

### Phase 2.5 - Dataset suitability check

- Goal: refine the dataset config, add pixel-projected size analysis at `imgsz=640`, and decide whether the current dataset is strong enough for the small-object research direction.

Files created:

- `configs/dataset/drill_bit_yolo.yaml`

Files modified:

- `src/xai_evidence_sod/data/validation.py`
- `src/xai_evidence_sod/data/bbox_stats.py`
- `src/xai_evidence_sod/data/dataset_report.py`
- `scripts/inspect_dataset.py`
- `docs/experiment_log.md`

Commands to run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/inspect_dataset.py --data configs/dataset/drill_bit_yolo.yaml --output artifacts/dataset_inspection --imgsz 640`

Expected outputs:

- Updated `bbox_size_stats.csv` with projected pixel fields
- Updated markdown and JSON summaries with normalized and pixel-projected size distributions
- A clearer dataset suitability conclusion before baseline training

Checks performed:

- Recompiled `src/` and `scripts/`
- Re-ran dataset inspection with the real dataset config at `imgsz=640`

Risks / open questions:

- Empty label files may be valid negatives, but a spot-check is still needed before training
- The dataset may be better for weak-evidence analysis than for strongly tiny-object-focused claims
- A more small-object-heavy dataset may still be needed later for stronger validation

### Phase 3 - Baseline training and evaluation

- Goal: build a stable Ultralytics YOLO baseline on the drill-bit dataset and export baseline evaluation artifacts before any XAI-guided method is introduced.

Files created:

- `src/xai_evidence_sod/utils/config.py`
- `src/xai_evidence_sod/models/yolo_wrapper.py`
- `src/xai_evidence_sod/evaluation/baseline_metrics.py`
- `configs/train/baseline_drill_bit_yolov8n.yaml`
- `scripts/evaluate_baseline.py`
- `docs/baseline_phase3_report.md`

Files modified:

- `src/xai_evidence_sod/models/__init__.py`
- `src/xai_evidence_sod/evaluation/__init__.py`
- `scripts/train_baseline.py`
- `scripts/eval_baseline.py`
- `docs/experiment_log.md`

Commands to run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/train_baseline.py --config configs/train/baseline_drill_bit_yolov8n.yaml`
- `PYTHONPATH=src .venv/bin/python scripts/evaluate_baseline.py --config configs/train/baseline_drill_bit_yolov8n.yaml --weights experiments/baseline_drill_bit/weights/best.pt`

Expected outputs:

- `experiments/baseline_drill_bit/`
- `artifacts/baseline_eval/metrics_overall.json`
- `artifacts/baseline_eval/baseline_eval_report.md`
- prediction sample images and `Chipped` error-case CSV

Checks performed:

- Dependency installation for `ultralytics` and related runtime packages
- `compileall` sanity check after script and helper creation
- optional smoke training may still depend on runtime budget and hardware

Risks / open questions:

- The dataset is not strongly tiny-object-focused, so baseline conclusions should be framed around weak-evidence and class-specific behavior rather than broad tiny-object claims
- `yolov8n.pt` may require external weight download on first use if not cached
- Full training time on CPU may be long even though smoke testing is possible

### Phase 3.5 - Baseline sanity and dataset spot-check

- Goal: verify empty-label images, sample baseline-ready error cases, and confirm full baseline commands/paths before any scientific interpretation.

Files created:

- `scripts/spot_check_empty_labels.py`
- `scripts/sample_error_cases.py`
- `docs/phase3_5_dataset_sanity.md`

Files modified:

- `docs/baseline_phase3_report.md`
- `docs/experiment_log.md`

Commands to run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/spot_check_empty_labels.py --data configs/dataset/drill_bit_yolo.yaml --output artifacts/dataset_spotcheck_empty --num-samples 64 --seed 0`
- `PYTHONPATH=src .venv/bin/python scripts/sample_error_cases.py --csv artifacts/baseline_eval_smoke/chipped_error_cases.csv --output artifacts/baseline_error_samples --num-samples 64 --seed 0`

Expected outputs:

- `artifacts/dataset_spotcheck_empty/selected_empty_labels.csv`
- `artifacts/dataset_spotcheck_empty/empty_label_contact_sheet.jpg`
- `artifacts/baseline_error_samples/sampled_error_cases.csv`

Checks performed:

- compile sanity after adding the scripts
- optional smoke spot-check run with a smaller sample count

Risks / open questions:

- Empty-label images may still hide annotation misses, so manual review remains necessary
- `Chipped` smoke false negatives are useful for readiness review but not for scientific claims
