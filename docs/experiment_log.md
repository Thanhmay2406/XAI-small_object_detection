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

### Phase 4 - Baseline error analysis and evidence preparation

- Goal: lock the full baseline result as the official comparison point, analyze where the baseline fails, and prepare reproducible artifacts for later XAI/evidence inspection.

Files created:

- `src/xai_evidence_sod/evaluation/error_analysis.py`
- `scripts/analyze_baseline_errors.py`
- `scripts/export_error_gallery.py`
- `docs/phase4_error_analysis_and_evidence_plan.md`

Files modified:

- `src/xai_evidence_sod/evaluation/__init__.py`
- `docs/baseline_phase3_report.md`
- `docs/experiment_log.md`

Commands to run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/analyze_baseline_errors.py --eval-dir artifacts/baseline_eval --data configs/dataset/drill_bit_yolo.yaml --output artifacts/baseline_error_analysis --focus-class Chipped`
- `PYTHONPATH=src .venv/bin/python scripts/export_error_gallery.py --csv artifacts/baseline_error_analysis/focus_class_error_cases.csv --output artifacts/baseline_error_gallery --num-samples 32 --seed 0`

Expected outputs:

- `artifacts/baseline_error_analysis/error_summary.json`
- `artifacts/baseline_error_analysis/per_class_error_summary.csv`
- `artifacts/baseline_error_analysis/focus_class_error_summary.json`
- `artifacts/baseline_error_analysis/focus_class_error_cases.csv`
- `artifacts/baseline_error_analysis/size_bin_error_summary.csv`
- `artifacts/baseline_error_analysis/confidence_error_summary.csv`
- `artifacts/baseline_error_gallery/sampled_errors.csv`
- `artifacts/baseline_error_gallery/error_gallery_contact_sheet.jpg`

Checks performed:

- `compileall` passed after adding the new evaluation helpers and scripts
- baseline error analysis ran successfully on `artifacts/baseline_eval`
- manual-review gallery export ran successfully on the `Chipped` focus-class CSV

Risks / open questions:

- Current error taxonomy is heuristic because exported artifacts do not yet include per-GT matching assignments, saliency maps, or feature activations
- The dataset should still be framed as weak-evidence / hard-class detection rather than a strongly tiny-object dataset
- `Chipped` remains the highest-priority class for Phase 5 evidence-map inspection, but some near-threshold and cross-class overlaps may also indicate annotation ambiguity

### Phase 5 - XAI evidence extraction on curated baseline cases

- Goal: run post-hoc XAI evidence extraction on a curated `Chipped` subset using the locked baseline checkpoint, without retraining or changing the detector.

Files created:

- `src/xai_evidence_sod/xai/case_selection.py`
- `src/xai_evidence_sod/xai/cam.py`
- `src/xai_evidence_sod/xai/evidence_metrics.py`
- `src/xai_evidence_sod/xai/evidence_pipeline.py`
- `scripts/extract_xai_evidence.py`
- `docs/phase5_xai_evidence_extraction.md`

Files modified:

- `src/xai_evidence_sod/xai/__init__.py`
- `docs/experiment_log.md`

Commands to run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/extract_xai_evidence.py --weights experiments/baseline_drill_bit/weights/best.pt --data configs/dataset/drill_bit_yolo.yaml --cases artifacts/baseline_error_analysis/focus_class_error_cases.csv --output artifacts/xai_evidence_chipped --focus-class Chipped --methods eigencam --max-cases 16 --seed 0`
- `PYTHONPATH=src .venv/bin/python scripts/extract_xai_evidence.py --weights experiments/baseline_drill_bit/weights/best.pt --data configs/dataset/drill_bit_yolo.yaml --cases artifacts/baseline_error_analysis/focus_class_error_cases.csv --output artifacts/xai_evidence_chipped --focus-class Chipped --methods eigencam --max-cases 64 --seed 0`

Expected outputs:

- `artifacts/xai_evidence_chipped/evidence_cases.csv`
- `artifacts/xai_evidence_chipped/evidence_summary.json`
- `artifacts/xai_evidence_chipped/overlays/`
- `artifacts/xai_evidence_chipped/crops/`
- `artifacts/xai_evidence_chipped/maps/`
- `artifacts/xai_evidence_chipped/contact_sheets/`
- `artifacts/xai_evidence_chipped/README.md`

Checks performed:

- `compileall` passed after adding the Phase 5 XAI modules and CLI
- smoke extraction with `--max-cases 16` passed
- full extraction with `--max-cases 64` passed

Risks / open questions:

- Only `EigenCAM` is implemented in Phase 5; `Grad-CAM` and `Grad-CAM++` are scaffolded but intentionally left unimplemented until the YOLO target interface is stabilized
- Evidence metrics are descriptive post-hoc summaries and should not be read as proof that the detector causally uses the highlighted pixels
- `true_positive_proxy` rows are reconstructed from exported predictions and labels rather than coming from a native validator match export

### Phase 6 - Evidence review and failure-mode comparison

- Goal: compare descriptive Phase 5 EigenCAM evidence patterns across curated `Chipped` case groups and export a manual-review shortlist for Phase 7 scoping, without retraining or changing the detector.

Files created:

- `src/xai_evidence_sod/xai/evidence_review.py`
- `scripts/review_xai_evidence.py`
- `docs/phase6_evidence_review.md`

Files modified:

- `src/xai_evidence_sod/xai/__init__.py`
- `docs/experiment_log.md`

Commands to run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/review_xai_evidence.py --evidence-csv artifacts/xai_evidence_chipped/evidence_cases.csv --output artifacts/xai_evidence_review_chipped --focus-class Chipped --top-k 8`

Expected outputs:

- `artifacts/xai_evidence_review_chipped/evidence_group_summary.csv`
- `artifacts/xai_evidence_review_chipped/evidence_group_summary.json`
- `artifacts/xai_evidence_review_chipped/representative_cases.csv`
- `artifacts/xai_evidence_review_chipped/review_notes_template.csv`
- `artifacts/xai_evidence_review_chipped/README.md`
- optional `artifacts/xai_evidence_review_chipped/evidence_group_means.png`
- optional `artifacts/xai_evidence_review_chipped/peak_inside_gt_rate.png`

Checks performed:

- `compileall` passed after adding the Phase 6 review module and CLI
- the requested review command ran successfully on `artifacts/xai_evidence_chipped/evidence_cases.csv`
- review artifacts were exported with relative paths and optional plots

Current quantitative snapshot:

- `false_negative` mean `energy_in_gt_box`: `0.0072`
- `false_negative` `peak_inside_gt_box_rate`: `0.0`
- `localization_error` mean `energy_in_gt_box`: `0.1432`
- `near_threshold_overlap` count: `24`
- representative review buckets exported: `7` buckets x `8` rows each

Risks / open questions:

- Phase 6 remains descriptive only and does not justify causal claims about detector reasoning
- the `true_positive_proxy` group is heterogeneous, with a low median GT-box energy despite a higher mean driven by outliers
- some `false_positive` rows retain high GT-box energy, so taxonomy and overlap structure still need manual review before Phase 7 hypotheses are tightened

### Phase 7 - Manual evidence review and intervention design

- Goal: standardize manual review for the Phase 6 representative `Chipped` evidence cases and define a Phase 8 intervention decision gate without training any intervention.

Files created:

- `src/xai_evidence_sod/xai/manual_review.py`
- `scripts/prepare_manual_evidence_review.py`
- `docs/phase7_manual_evidence_review.md`

Files modified:

- `src/xai_evidence_sod/xai/__init__.py`
- `docs/experiment_log.md`

Commands to run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_manual_evidence_review.py --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv --output artifacts/manual_evidence_review_chipped --focus-class Chipped`

Expected outputs:

- `artifacts/manual_evidence_review_chipped/manual_review_template.csv`
- `artifacts/manual_evidence_review_chipped/manual_review_guide.md`
- `artifacts/manual_evidence_review_chipped/intervention_decision_table.md`
- `artifacts/manual_evidence_review_chipped/README.md`
- optional `artifacts/manual_evidence_review_chipped/manual_review_summary.csv`
- optional `artifacts/manual_evidence_review_chipped/manual_review_summary.json`

Checks performed:

- `compileall` passed after adding the Phase 7 manual-review module and CLI
- the requested Phase 7 preparation command ran successfully
- the output template was exported with relative paths and controlled review fields

Current run status:

- `manual_review_template.csv` contains the requested review columns plus Phase 6-derived metadata such as `bucket`, `error_type`, and `tags`
- `manual_review_guide.md` records the review workflow and allowed categorical values
- `intervention_decision_table.md` defines the Phase 8 gate for `hard_sample_weighting`, `chipped_focused_augmentation`, `background_negative_mining`, `label_review`, `cross_method_xai_check`, and `no_action`
- no `manual_review_summary.*` files were produced yet because `artifacts/manual_evidence_review_chipped/manual_review_filled.csv` does not exist at this point

Risks / open questions:

- Phase 7 remains workflow and hypothesis preparation only; it does not validate any intervention
- manual review quality will depend on consistent use of the controlled fields across buckets
- if many rows end up marked `cam_method_uncertain` or `uncertain`, a cross-method post-hoc check may be needed before Phase 8

### Phase 8 - Evidence-to-intervention decision design

- Goal: translate Phase 6 evidence buckets and Phase 7 review infrastructure into a conservative intervention-design layer without running training or changing model/XAI code.

Files created:

- `src/xai_evidence_sod/xai/intervention_design.py`
- `scripts/design_phase8_interventions.py`
- `docs/phase8_intervention_design.md`

Files modified:

- `src/xai_evidence_sod/xai/__init__.py`
- `docs/experiment_log.md`

Commands run:

- `PYTHONPATH=src .venv/bin/python scripts/design_phase8_interventions.py --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv --manual-review-summary artifacts/manual_evidence_review_chipped/manual_review_summary.csv --manual-review-filled artifacts/manual_evidence_review_chipped/manual_review_filled.csv --output artifacts/intervention_design_chipped --focus-class Chipped`
- `.venv/bin/python -m compileall src scripts`

Outputs created:

- `artifacts/intervention_design_chipped/intervention_decision_table.csv`
- `artifacts/intervention_design_chipped/intervention_decision_table.json`
- `artifacts/intervention_design_chipped/intervention_candidates.csv`
- `artifacts/intervention_design_chipped/intervention_candidates.json`
- `artifacts/intervention_design_chipped/no_intervention_or_insufficient_evidence.csv`
- `artifacts/intervention_design_chipped/README.md`

Current run status:

- the decision-design script completed successfully on the current artifacts
- the output README records `manual_review_source = demo_or_synthetic`
- `manual_review_used_as_research_evidence = false` is enforced in the generated JSON/README
- most buckets remain `NO_INTERVENTION_YET`, with only cautious Phase 9 design candidates exported for `DATA_SAMPLING_OR_CURRICULUM` and `SALIENCY_GUIDED_ATTENTION_PROTOTYPE`
- no training, architecture changes, loss changes, or XAI extraction changes were introduced in this phase

Risks / open questions:

- current manual-review inputs are smoke-demo artifacts and cannot be used as real research evidence
- candidate interventions remain hypothesis-level only until real manual review is completed
- false-positive and near-threshold buckets still need better disambiguation before any intervention is justified

### Phase 8.5 - Real manual review completion support

- Goal: prepare a clean real-review handoff so the synthetic Phase 7 smoke review can be replaced by real manual annotation without changing training, architecture, loss, or XAI extraction code.

Files created:

- `docs/phase8_5_real_manual_review.md`
- `scripts/validate_manual_review_real.py`

Outputs created:

- `artifacts/manual_evidence_review_chipped/manual_review_real_template.csv`
- `artifacts/manual_evidence_review_chipped/real_manual_review_checklist.md`
- `artifacts/manual_evidence_review_chipped/manual_review_real_validation.json`

Commands run:

- `PYTHONPATH=src .venv/bin/python scripts/validate_manual_review_real.py --manual-review artifacts/manual_evidence_review_chipped/manual_review_filled.csv --output artifacts/manual_evidence_review_chipped/manual_review_real_validation.json`
- `.venv/bin/python -m compileall src scripts`

Current run status:

- `manual_review_filled.csv` is still detected as `demo_or_synthetic`
- validation failed intentionally because synthetic markers remain in `reviewer_notes`
- `manual_review_used_as_research_evidence = false` remains the correct state
- a clean `manual_review_real_template.csv` is now available for real review completion
- Phase 7 summary and Phase 8 decision design were not rerun because no real review has been completed yet

Risks / open questions:

- replacing `manual_review_filled.csv` should be done only after the real template is actually reviewed
- the highest-priority buckets remain `localization_misaligned_evidence` and `near_threshold_high_evidence`
- Phase 9 is still blocked until real review validation passes and downstream summaries are regenerated

## 2026-06-27

### Phase 11F.1 - Kaggle execution adapter package preparation

- Scope: prepare-only Kaggle adapter phase built from the already prepared Phase 11F execution package.
- Added `scripts/prepare_phase11f_1_kaggle_execution_adapter_package.py` to confirm the Phase 11F gate state, read the prepared Phase 11F command, and generate a Kaggle-safe parameterized command package without changing approved training semantics.
- Added `docs/phase11f_1_kaggle_execution_adapter_package.md` and a new adapter bundle under `artifacts/phase11f_1_kaggle_execution_adapter_package/`.
- Kept the phase non-executing and non-mutating: no training, evaluation, or inference ran; no dataset was mutated; and the Phase 11F summary was treated as read-only input.
- Preserved the core training parameters from Phase 11F while replacing local absolute paths with Kaggle-oriented variables.
- Next step: `phase11g_execute_approved_staging_training_on_kaggle`

### Phase 11F - Approved staging training execution package preparation

- Scope: prepare-only execution-package phase after Phase 11E real approval validation.
- Added `scripts/prepare_phase11f_approved_staging_training_execution_package.py` to validate the Phase 11E pass state, freeze the approved staging-training inputs and command, and emit a non-execution package for later use.
- Added `docs/phase11f_approved_staging_training_execution_package.md` and a new package bundle under `artifacts/phase11f_approved_staging_training_execution_package/`.
- Kept the phase non-executing and non-mutating: training, evaluation, and inference were not executed; original and staging datasets were not mutated; and historical Phase 9Z.5 state remained unchanged.
- Phase 11E validated the real approval path, so the execution package is now prepared for later approved use.
- Next step: `phase11g_execute_approved_staging_training`

### Phase 11E.1 - Real approval validator logic review and conservative patch

- Scope: logic-review-only patch for the Phase 11E-real approval validator.
- Patched `scripts/validate_phase11e_real_human_training_approval.py` so historical Phase 9Z.5 blocked state is preserved as context rather than treated as an automatic rejection of newly valid approval content.
- Fixed two conservative validator issues: explicit negative wording like `original dataset labels must not be mutated` no longer counts as authorization, and template-comparison logic no longer compares the filled approval file against itself.
- Added `docs/phase11e_1_real_approval_validator_logic_patch.md` and a new review bundle under `artifacts/phase11e_1_real_approval_validator_logic_patch/`.
- Expected validator result after patch: `phase11e_real_human_training_approval_validated`
