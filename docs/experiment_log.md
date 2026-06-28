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

## 2026-06-28

### Phase 11R.1 - Filled checkpoint publication decision validation

- Scope: strict validation-only follow-up to Phase 11R using the human-filled publication decision CSV already emitted by the metadata-only gate.
- Added `scripts/validate_phase11r1_checkpoint_publication_decision.py` to read the Phase 11R summary, checkpoint inventory, and decision template, then validate the filled decision by content.
- Added `docs/phase11r1_checkpoint_publication_decision_validation.md`.
- Added a new artifact bundle under `artifacts/phase11r1_checkpoint_publication_decision_validation/`.
- Phase 11R.1 does not load checkpoints with `torch`, `ultralytics`, or any model library.
- Phase 11R.1 does not copy or upload checkpoints and does not execute publication.
- Phase 11R.1 does not run training, evaluation, inference, prediction, or export.
- Phase 11R.1 does not mutate datasets, labels, YAMLs, or other approval-state inputs.
- Blocking conditions include `pending_manual_decision`, empty reviewer identity, empty publication target, and missing metric caveat acknowledgment.
- If the human decision is `approve_publication_package_preparation`, Phase 11R.1 allows only the next package-preparation phase and still keeps `checkpoint_upload_executed = false`.

### Phase 11R - Checkpoint publication decision gate

- Scope: metadata-only decision gate for whether and how the Phase 11 training checkpoint should be published.
- Added `scripts/prepare_phase11r_checkpoint_publication_decision_gate.py` to inventory likely checkpoint files by metadata only, emit a publication decision template, and carry forward the Phase 11P report/caveat status without performing any upload.
- Added `docs/phase11r_checkpoint_publication_decision_gate.md`.
- Added a new artifact bundle under `artifacts/phase11r_checkpoint_publication_decision_gate/`.
- Phase 11R does not load checkpoints, does not copy checkpoints into artifacts, and does not upload to GitHub, Git LFS, GitHub Releases, Kaggle, or external storage.
- Phase 11R does not run training, evaluation, inference, prediction, or export.
- Default outcome without a filled decision CSV is `status = phase11r_checkpoint_publication_decision_pending`.
- Default next allowed step is `fill_phase11r_publication_decision_template_then_rerun_phase11r`.

### Phase 11Q - Scoped git commit and repository handoff audit for Phase 11N-11P

- Scope: read-only git handoff audit before any scoped `git add` / `git commit` / `git push`.
- Added `scripts/audit_phase11q_scoped_git_handoff.py` to inspect `git status --short`, classify exact Phase 11N-11P candidates, separate out-of-scope dirty files, detect blocked large/binary/runtime outputs, and prepare command previews without executing them.
- Added `docs/phase11q_scoped_git_handoff_audit.md`.
- Added a new artifact bundle under `artifacts/phase11q_scoped_git_handoff_audit/`.
- Phase 11Q does not run `git add`, `git commit`, or `git push`.
- Phase 11Q does not run training, evaluation, inference, prediction, or export.
- Phase 11Q does not load checkpoints, mutate datasets, mutate labels, or copy large outputs into artifacts.
- The purpose of this phase is only to prepare a safe scoped handoff manifest and preview commands for later manual review.

### Phase 11P - Final test evaluation evidence report with metric provenance caveat

- Scope: strict report-only consolidation phase for Phase 11J through Phase 11O.
- Added `scripts/prepare_phase11p_final_test_evaluation_report.py` to read the Phase 11N and Phase 11O summaries, prepare a final markdown report, and emit a metric provenance table without recomputing metrics.
- Added `docs/phase11p_final_test_evaluation_report_with_caveat.md`.
- Added a new artifact bundle under `artifacts/phase11p_final_test_evaluation_report/`.
- Phase 11P explicitly does not run training, evaluation, inference, prediction, or export.
- Phase 11P does not load checkpoints, does not parse `predictions.json` for metric recomputation, and does not copy large outputs into artifacts.
- If Phase 11O reporting remains blocked, Phase 11P prepares a caveated final report that does not claim numeric aggregate test metrics.
- If Phase 11O later unlocks reporting, Phase 11P may include manually validated test metrics with explicit manual-provenance caveat.

### Phase 11O - Manual test metric review gate

- Scope: strict review-only manual metric provenance gate after Phase 11N flagged that credible test evaluation outputs exist but aggregate test metrics still require manual extraction.
- Added `scripts/validate_phase11o_manual_test_metric_review.py` to read the Phase 11N summary, emit a manual review template, validate exactly one manually entered review row, and gate later reporting without recomputing metrics.
- Added `docs/phase11o_manual_test_metric_review_gate.md`.
- Added a new artifact bundle under `artifacts/phase11o_manual_test_metric_review/`.
- Phase 11O explicitly does not run training, evaluation, inference, prediction, or export.
- Phase 11O does not mutate datasets, YAML files, labels, or images.
- Phase 11O does not load checkpoints, parse `predictions.json` to recompute metrics, or copy large runtime outputs into artifacts.
- Default outcome without a valid filled review CSV is `status = phase11o_manual_test_metric_review_pending`.
- Default next allowed step is `fill_phase11o_manual_metric_review_csv_from_kaggle_visible_output_then_rerun_phase11o`.

### Phase 11N - Test evaluation output collection and validation

- Scope: strict non-execution collector/validator for existing Phase 11M.1 test evaluation outputs.
- Added `scripts/collect_phase11n_test_evaluation_outputs.py` to inspect and validate the existing evaluation output directory and the related training output directory without executing anything.
- Added `docs/phase11n_test_evaluation_output_collection_and_validation.md`.
- Phase 11N inspects:
  - `experiments/phase11m_test_eval/yolov8n_drill_bit_phase11m_test_eval`
  - `experiments/phase11j_training`
- Phase 11N does not run evaluation, inference, prediction, training, or export.
- Phase 11N does not mutate datasets or labels.
- Phase 11N does not load, copy, or modify checkpoint weights.
- Phase 11N does not copy large evaluation outputs into artifacts.
- Resulting status: `phase11n_test_evaluation_outputs_collected_needs_manual_metric_review`.
- Next allowed step: `manually_extract_phase11m1_metrics_then_rerun_phase11n_or_continue_with_caveat`.

### Phase 11M.1 - Explicit approved test evaluation execution wrapper

- Scope: explicit execution wrapper for the Phase 11M.0 locked test evaluation package.
- Added `scripts/execute_phase11m1_approved_test_evaluation.py` to validate the Phase 11M.0 package, support Kaggle GPU runtime path overrides, prepare the exact runtime `yolo detect val` command, and execute it only when explicitly allowed.
- Added `docs/phase11m1_approved_test_evaluation_execution.md`.
- Phase 11M.1 is blocked by default and carries forward the existing Phase 11J.1 / Phase 11K / Phase 11L provenance caveat.
- Phase 11M.1 does not train, mutate datasets, mutate labels, copy or modify weights, or load checkpoint tensors directly in Python.
- Local verification was run only in default non-execution mode, so evaluation was not executed.
- Resulting status: `phase11m1_test_evaluation_blocked_missing_execute_or_approval`.
- Next allowed step: `provide_explicit_execute_flag_or_filled_phase11m1_approval_csv`.

### Phase 11 Runtime Output Relocation

- Moved the local Phase 11 training and evaluation runtime output roots out of the repo top level and under `experiments/` to keep source/docs and runtime outputs separated more cleanly.
- Updated the Phase 11K default training-output inspection path to `experiments/phase11j_training/yolov8n_drill_bit_phase11j`.
- Updated the Phase 11M.0 and Phase 11M.1 repo-local evaluation project path to `experiments/phase11m_test_eval`.
- This relocation is organizational only and does not run training, evaluation, inference, dataset mutation, or checkpoint modification.

### Phase 11M.0 - Prepare-only approved test evaluation package

- Scope: strict prepare-only and non-execution phase after the passed Phase 11L checkpoint integrity gate.
- Added `scripts/prepare_phase11m0_approved_test_evaluation_no_execution.py` to validate the Phase 11L summary, verify the accepted `best.pt` by metadata only, resolve the dataset YAML path conservatively, and prepare a locked `yolo detect val` command without executing it.
- Added `docs/phase11m0_prepare_approved_test_evaluation_no_execution.md`.
- Phase 11M.0 created the evaluation approval template for Phase 11M.1 and carried forward the existing Phase 11J.1 / Phase 11K / Phase 11L provenance caveat.
- Phase 11M.0 did not run evaluation, inference, prediction, training, export, dataset mutation, label mutation, checkpoint loading, or weight copying.
- Resulting status: `phase11m0_approved_test_evaluation_prepare_only_passed`.
- Next allowed step: `collect_phase11m1_explicit_evaluation_execution_approval_or_execute_with_flag`.

### Phase 11L - Training output integrity and provenance validation

- Scope: strict non-execution validation phase for the Phase 11K collected training outputs.
- Added `scripts/validate_phase11l_training_output_integrity_and_provenance.py` to validate the Phase 11K artifacts, re-parse the local `results.csv`, compare direct metrics against the Phase 11K summary, and re-check checkpoint metadata by path, size, mtime, and streamed `sha256` only.
- Added `docs/phase11l_training_output_integrity_and_provenance.md`.
- Phase 11L records the existing provenance caveat that the repo-local Phase 11J.1 summary still reports `phase11j1_execution_not_started_missing_execute_flag`, so this phase validates only local output integrity and not Kaggle execution provenance.
- Phase 11L does not train, evaluate, infer, mutate datasets, load checkpoint tensors, or copy weights into artifacts.
- Resulting status: `phase11l_training_output_integrity_and_provenance_passed`.
- Next allowed step: `phase11m0_prepare_approved_test_evaluation_no_execution`.

### Phase 11K - Training outputs and metrics collection

- Scope: strict non-execution output-collection and provenance-materialization phase after a completed Kaggle Phase 11J.1 run was copied back locally.
- Added `scripts/collect_phase11k_training_outputs_and_metrics.py` to inspect the existing local YOLO run directory, hash the existing `best.pt` and `last.pt` files by streamed `sha256`, parse `results.csv`, and emit only small metadata artifacts.
- Added `docs/phase11k_training_outputs_and_metrics.md`.
- Inspected local training output directory: `phase11j_training/yolov8n_drill_bit_phase11j`.
- Found `results.csv`, `weights/best.pt`, `weights/last.pt`, and `args.yaml`.
- Because the repo-local Phase 11J.1 summary still shows the earlier dry-run state, Phase 11K conservatively recorded `phase11j1_summary_available = false` and relied on direct local output inspection instead of claiming success from that summary.
- Parsed 100 `results.csv` rows.
- Best tracked metric used `metrics/mAP50-95(B)` and selected epoch 42 with `0.36688`.
- Final epoch 100 metrics included `metrics/mAP50-95(B) = 0.35709` and `metrics/mAP50(B) = 0.71681`.
- No training, evaluation, inference, dataset mutation, Kaggle upload, or weight creation/copying occurred in Phase 11K.
- Next step: `phase11l_evaluate_trained_model_on_approved_test_split`

### Phase 11J.1 - Locked Kaggle training execution wrapper

- Scope: controlled execution-wrapper phase for the exact Phase 11J.0 locked Kaggle command.
- Added `scripts/execute_phase11j1_locked_kaggle_training.py` to validate the Phase 11J.0 lock, stay in local dry-run mode by default, and execute only with `--execute` in a Kaggle runtime.
- Added `docs/phase11j1_locked_kaggle_training_execution.md`.
- Local verification remains non-executing and should produce `status = phase11j1_execution_not_started_missing_execute_flag`.
- Actual Kaggle execution is separate from local preparation and should only occur when `--execute` is supplied on Kaggle.
- Weights creation remains `false` in local dry-run preparation and becomes runtime-dependent only after real Kaggle execution.

### Phase 11J.0 - Approved Kaggle training command lock

- Scope: strict non-execution command-lock and handoff-record phase after a passed Phase 11I human training approval gate.
- Added `scripts/prepare_phase11j0_approved_kaggle_training_command_lock.py` to validate a passed Phase 11I gate plus the real approval CSV and lock the exact approved Kaggle training command without modifying it.
- Added `docs/phase11j0_approved_kaggle_training_command_lock.md`.
- Phase 11J.0 is defined to keep training, evaluation, inference, dataset mutation, Kaggle upload, and weights/checkpoint creation all at `false`.
- Expected next step after a valid command lock: `phase11j1_execute_locked_kaggle_training`

### Phase 11I - Human training execution approval gate

- Scope: strict gate-only and record-only human approval phase after the Phase 11H Kaggle manual preflight pass.
- Added `scripts/prepare_phase11i_human_training_execution_approval_gate.py` to validate Phase 11H summary/gate outputs, generate a human approval template and checklist, and emit a blocked-by-default approval gate record.
- Added `docs/phase11i_human_training_execution_approval_gate.md` and a new bundle under `artifacts/phase11i_human_training_execution_approval_gate/`.
- Default outcome without a real approval CSV is `status = phase11i_blocked_waiting_human_training_execution_approval`.
- `training_allowed = false`, `ready_for_training_execution = false`, and human approval remains required.
- No training, evaluation, or inference was executed; no dataset mutation, Kaggle upload, weights, or checkpoints were created.
- Next step: `collect_real_human_training_execution_approval`

### Phase 11H - Kaggle manual preflight validation materialization

- Scope: gate-only and record-only materialization of a copied Kaggle manual preflight summary.
- Added `scripts/materialize_phase11h_kaggle_manual_preflight_validation.py` to validate the copied Phase 11H input summary, re-check split and YAML fields from that summary, and emit a conservative artifact bundle without executing anything.
- Added `docs/phase11h_kaggle_manual_preflight_validation.md` and a new bundle under `artifacts/phase11h_kaggle_manual_preflight_validation/`.
- Kaggle manual preflight passed structurally with split counts `train=4219`, `valid=916`, `test=906`.
- No missing, orphan, or invalid labels were reported in any split.
- `ready_for_training_execution_candidate = true`, but `ready_for_training_execution = false` remains locked.
- Training approval is still required before any later execution phase.
- No training, evaluation, or inference was executed; no dataset mutation, Kaggle upload, weights, or checkpoints were created.
- Next step: `phase11i_human_training_execution_approval_gate`

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
- Added `scripts/review_patch_phase11e_real_approval_validator_logic.py` and patched `scripts/validate_phase11e_real_human_training_approval.py` so historical Phase 9Z.5 blocked state is preserved as context, not treated as an automatic rejection of newly valid approval content.
- Fixed two conservative validator issues: explicit negative wording like "original dataset labels must not be mutated" no longer counts as authorization, and template-comparison logic no longer compares the filled approval file against itself.
- Added `docs/phase11e_1_real_approval_validator_logic_patch.md` and a new review bundle under `artifacts/phase11e_1_real_approval_validator_logic_patch/`.
- Expected validator result after patch: `phase11e_real_human_training_approval_validated`

### Phase 11E-real - Real human training approval validation

- Scope: approval-content-validation-only phase before any later training-preparation phase may be created.
- Added `scripts/validate_phase11e_real_human_training_approval.py` to validate a candidate approval CSV by contents, reject placeholders/example provenance or ambiguous scope, and preserve the non-executing boundary.
- Added `docs/phase11e_real_human_training_approval_validation.md` and a new validation bundle under `artifacts/phase11e_real_approval_validation/`.
- Kept the phase conservative: no training is executed here, no pending approval becomes valid by filename alone, and any pass only enables a later preparation phase rather than execution.
- Expected pass status: `phase11e_real_human_training_approval_validated`
- Expected fail status: `phase11e_real_human_training_approval_rejected_or_incomplete`

### Phase 11D - Real-approval collection and no-training paper-finalization gate

- Scope: decision-materials-only phase that prepares either a real human approval collection branch or an audit-only paper-finalization branch without changing any approval state.
- Added `scripts/prepare_phase11d_approval_or_no_training_gate.py` to generate the approval checklist/template, approval evidence requirements, training-allowed conditions, audit-only allowed/forbidden claims, figures/tables possible without training, and limitations draft points.
- Added `docs/phase11d_approval_or_no_training_gate.md` and a new gate bundle under `artifacts/phase11d_approval_or_no_training_gate/`.
- Kept the gate conservative: filename or path naming is not approval, approval must be validated from file contents in a later phase, and `training_allowed_after_phase11d` remains false.
- Expected final status: `phase11d_approval_or_no_training_gate_prepared`
- Expected next step: `collect_real_human_training_approval_or_finalize_audit_only_paper`

### Phase 11C - Paper direction and experiment-plan preparation

- Scope: planning-only and documentation-only phase derived from the Phase 11B research-state analysis.
- Added `scripts/prepare_phase11c_paper_direction_and_experiment_plan.py` to produce a conservative paper-direction package, safe claim set, claim-avoidance register, non-training experiment plan, conditional post-approval experiment plan, paper outline, and figures/tables plan.
- Added `docs/phase11c_paper_direction_and_experiment_plan.md` and a new planning bundle under `artifacts/phase11c_paper_direction_experiment_plan/`.
- Kept the framing conservative: XAI is diagnostic support, the one materialized relabel patch remains case-level only, and no training-oriented next step is allowed while Phase 9Z.5 approval is still invalid.
- Expected final status: `phase11c_paper_direction_and_experiment_plan_prepared`
- Expected next step: `phase11d_prepare_real_approval_collection_or_finalize_no_training_paper_plan`

### Phase 11B - Research-state analysis and next-direction decision package

- Scope: analysis-only and decision-preparation-only phase built from the corrected Phase 11A.1 resume context.
- Added `scripts/analyze_phase11b_research_state.py` to summarize the current research state, baseline performance, Chipped weakness, annotation uncertainty, XAI limitations, manual-review outcomes, and the Phase 9S through 9Z.5 gate chain.
- Added `docs/phase11b_research_state_analysis.md` and a new analysis bundle under `artifacts/phase11b_research_state_analysis/`.
- Kept the recommendation conservative: primary path is annotation correction / data-quality-first, fallback path is stop/pivot if evidence remains insufficient, and no training-oriented next step is allowed while Phase 9Z.5 approval is still unvalidated.
- Expected final status: `phase11b_research_state_analysis_completed`
- Expected next step: `phase11c_prepare_paper_direction_and_experiment_plan_no_training`

### Phase 11A.1 - Resume metadata and brief correction pass

- Scope: metadata-only correction pass for the existing Phase 11A research resume context package.
- Added `scripts/patch_phase11a_resume_metadata_and_brief.py` to validate the current Phase 11A package, recompute manifest and zip fields from disk, patch the research brief to prefer the full baseline metrics, and rebuild the Phase 11A zip with corrected embedded files.
- Added `docs/phase11a_1_resume_metadata_patch.md` and a new audit bundle under `artifacts/phase11a_1_resume_metadata_patch/`.
- Preserved the Phase 9Z.5 blocker with `training_allowed = false` and `approval_validated = false`.
- Expected final status: `phase11a_1_resume_metadata_patch_applied`
- Expected next step: `phase11b_research_state_analysis_from_corrected_resume_context`

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

### Phase 8.5 - Post-confirmation validation and rerun

- Goal: validate the confirmed manual review file, rerun Phase 7 summary, rerun Phase 8 decision design, and reassess whether Phase 9 can even be considered.

Files modified:

- `docs/experiment_log.md`

Commands run:

- `PYTHONPATH=src .venv/bin/python scripts/validate_manual_review_real.py --manual-review artifacts/manual_evidence_review_chipped/manual_review_filled.csv --output artifacts/manual_evidence_review_chipped/manual_review_real_validation.json`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_manual_evidence_review.py --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv --output artifacts/manual_evidence_review_chipped --focus-class Chipped`
- `PYTHONPATH=src .venv/bin/python scripts/design_phase8_interventions.py --representatives artifacts/xai_evidence_review_chipped/representative_cases.csv --group-summary artifacts/xai_evidence_review_chipped/evidence_group_summary.csv --manual-review-summary artifacts/manual_evidence_review_chipped/manual_review_summary.csv --manual-review-filled artifacts/manual_evidence_review_chipped/manual_review_filled.csv --output artifacts/intervention_design_chipped --focus-class Chipped`
- `.venv/bin/python -m compileall src scripts`

Outputs updated:

- `artifacts/manual_evidence_review_chipped/manual_review_real_validation.json`
- `artifacts/manual_evidence_review_chipped/manual_review_summary.csv`
- `artifacts/manual_evidence_review_chipped/manual_review_summary.json`
- `artifacts/intervention_design_chipped/intervention_decision_table.csv`
- `artifacts/intervention_design_chipped/intervention_decision_table.json`
- `artifacts/intervention_design_chipped/intervention_candidates.csv`
- `artifacts/intervention_design_chipped/intervention_candidates.json`
- `artifacts/intervention_design_chipped/no_intervention_or_insufficient_evidence.csv`
- `artifacts/intervention_design_chipped/README.md`

Current run status:

- validator passed on the confirmed `manual_review_filled.csv`
- Phase 7 summary reran successfully with `reviewed_row_count = 56` and `completed_row_count = 16`
- Phase 8 decision design reran successfully and exported `7` decision rows plus `2` candidate rows
- Phase 8 provenance gating was then fixed narrowly so the decision outputs now align with the validator and report `manual_review_source = real_candidate`
- `manual_review_used_as_research_evidence = true` is now reflected in the rerun Phase 8 JSON outputs
- no training, architecture, loss, or XAI extraction changes were made in this rerun step

Risks / open questions:

- only `16/56` rows are fully reviewed, so the evidence base is still partial even though the confirmed subset is now accepted as research evidence
- the current rerun exports only `2` candidate families and still leaves `5` evidence groups at `NO_INTERVENTION_YET`
- Phase 9 should still be treated as a decision under review rather than automatically unblocked, because the current intervention evidence remains narrow and bucket-specific

### Phase 9 - Readiness and prototype planning gate

- Goal: determine whether provenance blockers are removed and whether Phase 9 may proceed at the level of prototype planning only, without starting intervention training.

Files created:

- `scripts/assess_phase9_readiness.py`
- `docs/phase9_readiness_and_prototype_plan.md`

Outputs created:

- `artifacts/phase9_readiness_chipped/phase9_readiness_report.json`
- `artifacts/phase9_readiness_chipped/phase9_readiness_report.csv`
- `artifacts/phase9_readiness_chipped/phase9_candidate_plan.csv`
- `artifacts/phase9_readiness_chipped/README.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/assess_phase9_readiness.py --decision-table artifacts/intervention_design_chipped/intervention_decision_table.csv --candidates artifacts/intervention_design_chipped/intervention_candidates.csv --manual-review-summary artifacts/manual_evidence_review_chipped/manual_review_summary.csv --validation-report artifacts/manual_evidence_review_chipped/manual_review_real_validation.json --output artifacts/phase9_readiness_chipped --focus-class Chipped`

Current run status:

- provenance blocker is removed
- evidence coverage is still limited because only `16/56` review rows are complete
- Phase 9 is allowed only as `prototype_planning_only`
- `phase9_training_ready = false`
- `DATA_AUDIT_OR_RELABEL` is the primary planning direction
- `DATA_SAMPLING_OR_CURRICULUM` remains secondary and hypothesis-only

Risks / open questions:

- the readiness gate does not justify intervention training
- the project still lacks broad support across the five evidence groups that remain `NO_INTERVENTION_YET`
- architecture, loss, and XAI extraction changes remain unsupported by the current evidence base

### Phase 9B - Manual data audit summary

- Goal: summarize the current confirmation state of Phase 9A manual data audit without changing labels, creating datasets, or starting training.

Files created:

- `scripts/confirm_phase9a_manual_data_audit.py`
- `docs/phase9b_manual_data_audit.md`

Outputs created:

- `artifacts/phase9b_manual_data_audit_chipped/phase9b_manual_audit_summary.json`
- `artifacts/phase9b_manual_data_audit_chipped/phase9b_manual_audit_summary.csv`
- `artifacts/phase9b_manual_data_audit_chipped/phase9b_reviewed_relabel_decisions.csv`
- `artifacts/phase9b_manual_data_audit_chipped/README.md`

Commands run:

- `PYTHONPATH=src .venv/bin/python scripts/confirm_phase9a_manual_data_audit.py --audit-candidates artifacts/phase9a_data_audit_chipped/phase9a_audit_candidates.csv --relabel-decisions artifacts/phase9a_data_audit_chipped/phase9a_relabel_decision_template.csv --phase9a-plan artifacts/phase9a_data_audit_chipped/phase9a_data_audit_plan.json --output artifacts/phase9b_manual_data_audit_chipped --focus-class Chipped`
- `.venv/bin/python -m compileall src scripts`

Current run status:

- Phase 9B is summary-only and does not relabel data
- Phase 9B script was rerun successfully on the current `phase9a_relabel_decision_template.csv`
- output Phase 9B was regenerated under `artifacts/phase9b_manual_data_audit_chipped`
- current `phase9b_status = manual_data_audit_incomplete`
- training remains disallowed
- automatic relabel remains disallowed
- the next allowed step is `continue_manual_data_audit`
- the next step depends entirely on whether users complete the manual relabel decision template

### Phase 9B.1 - Assistant visual audit import

- Goal: import assistant visual audit outputs into the artifact tree for manual user review without changing any original annotation or treating assistant suggestions as confirmed audit results.

Files created:

- `artifacts/phase9b_assistant_review_outputs_chipped/README.md`
- `artifacts/phase9b_assistant_review_outputs_chipped/user_confirmation_checklist.md`

Files modified:

- `docs/experiment_log.md`

Outputs created:

- `artifacts/phase9b_assistant_review_outputs_chipped/phase9a_relabel_decision_template_assistant_suggested_pending_confirmation.csv`
- `artifacts/phase9b_assistant_review_outputs_chipped/phase9b_assistant_visual_audit_suggestions.csv`
- `artifacts/phase9b_assistant_review_outputs_chipped/phase9b_assistant_visual_audit_report.md`
- `artifacts/phase9b_assistant_review_outputs_chipped/phase9b_contact_sheet.jpg`
- `artifacts/phase9b_assistant_review_outputs_chipped/phase9b_zoom_contact_sheet.jpg`
- `artifacts/phase9a_data_audit_chipped/phase9a_relabel_decision_template_assistant_suggested_pending_confirmation.csv`

Commands run:

- `unzip -o phase9b_assistant_review_outputs.zip -d artifacts/phase9b_assistant_review_outputs_chipped`
- `cp artifacts/phase9b_assistant_review_outputs_chipped/phase9a_relabel_decision_template_assistant_suggested_pending_confirmation.csv artifacts/phase9a_data_audit_chipped/phase9a_relabel_decision_template_assistant_suggested_pending_confirmation.csv`
- `.venv/bin/python -m compileall src scripts`

Current run status:

- assistant visual audit outputs were imported into the artifact folder
- a manual confirmation checklist was created for all 8 Phase 9A audit cases
- provenance was not changed to `user_confirmed_manual_audit`
- `artifacts/phase9a_data_audit_chipped/phase9a_relabel_decision_template.csv` was not overwritten
- Phase 9B still requires manual user confirmation
- training remains locked

### Phase 9B.2 - Manual audit evidence package

- Goal: build a complete manual-only evidence package for the 8 Phase 9B audit candidates so the cases can be reviewed visually without training, relabeling, or modifying the dataset.

Files created:

- `scripts/package_phase9b_manual_audit_evidence.py`

Files modified:

- `docs/experiment_log.md`

Outputs created:

- `artifacts/phase9b_manual_audit_evidence_package_chipped/README.md`
- `artifacts/phase9b_manual_audit_evidence_package_chipped/package_manifest.csv`
- `artifacts/phase9b_manual_audit_evidence_package_chipped/package_manifest.json`
- `artifacts/phase9b_manual_audit_evidence_package_chipped/class_mapping.json`
- `artifacts/phase9b_manual_audit_evidence_package_chipped/dataset_config_copy.yaml`
- `artifacts/phase9b_manual_audit_evidence_package_chipped/annotation_guideline_candidates/`
- `artifacts/phase9b_manual_audit_evidence_package_chipped/contact_sheets/`
- `artifacts/phase9b_manual_audit_evidence_package_chipped/cases/`
- `artifacts/phase9b_manual_audit_evidence_package_chipped.zip`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/package_phase9b_manual_audit_evidence.py --audit-candidates artifacts/phase9a_data_audit_chipped/phase9a_audit_candidates.csv --relabel-template artifacts/phase9a_data_audit_chipped/phase9a_relabel_decision_template.csv --dataset-yaml configs/dataset/drill_bit_yolo.yaml --output artifacts/phase9b_manual_audit_evidence_package_chipped --focus-class Chipped`
- `cd artifacts && zip -r phase9b_manual_audit_evidence_package_chipped.zip phase9b_manual_audit_evidence_package_chipped`

Current run status:

- packaged `8` audit cases
- original images found for `8/8` cases
- labels found for `8/8` cases
- GT overlays created for `8/8` cases
- prediction overlays found for `8/8` cases
- XAI overlays found for `8/8` cases
- annotation guideline candidates were found and copied into the package
- training remains locked
- auto relabel remains locked
- original annotations were not modified
- dataset contents were not modified
- the next step is manual visual analysis and user confirmation

## 2026-06-25 - Phase 9B user confirmation completed

Files created:

- `artifacts/phase9a_data_audit_chipped/phase9b_user_confirmation_note.md`
- `artifacts/phase9b_manual_audit_analysis_outputs_chipped/`
- `artifacts/phase9a_data_audit_chipped/phase9a_relabel_decision_template.before_phase9b_user_confirmation.csv`

Files modified:

- `artifacts/phase9a_data_audit_chipped/phase9a_relabel_decision_template.csv`
- `artifacts/phase9b_manual_data_audit_chipped/README.md`
- `artifacts/phase9b_manual_data_audit_chipped/phase9b_manual_audit_summary.csv`
- `artifacts/phase9b_manual_data_audit_chipped/phase9b_manual_audit_summary.json`
- `artifacts/phase9b_manual_data_audit_chipped/phase9b_reviewed_relabel_decisions.csv`
- `docs/experiment_log.md`

Confirmation summary:

- `8/8` Phase 9B cases were user-confirmed through manual audit.
- `6` cases were confirmed as `keep_original_label`.
- `2` cases were confirmed as `needs_second_review`.
- `0` cases were confirmed as `relabel_needed`.
- `0` cases were confirmed as `exclude_from_training_candidate`.
- `phase9b_status = manual_data_audit_completed`
- training remains locked
- auto relabel remains locked
- next step: `Phase 9C - Manual Audit Outcome Summary`

Guardrails preserved:

- no automatic relabeling was applied
- original annotations were not modified
- no training was run
- no architecture changes were made
- no loss changes were made

## 2026-06-25 - Phase 9C manual audit outcome summarized

Files created:

- `scripts/summarize_phase9c_manual_audit_outcome.py`
- `docs/phase9c_manual_audit_outcome.md`
- `artifacts/phase9c_manual_audit_outcome_chipped/phase9c_case_outcomes.csv`
- `artifacts/phase9c_manual_audit_outcome_chipped/phase9c_second_review_candidates.csv`
- `artifacts/phase9c_manual_audit_outcome_chipped/phase9c_manual_audit_outcome_summary.json`
- `artifacts/phase9c_manual_audit_outcome_chipped/phase9c_manual_audit_outcome_summary.csv`
- `artifacts/phase9c_manual_audit_outcome_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- Phase 9C summary ran for output path `artifacts/phase9c_manual_audit_outcome_chipped`
- `phase9c_status = manual_audit_outcome_summarized`
- `keep_original_label = 6`
- `needs_second_review = 2`
- `relabel_needed = 0`
- `exclude_from_training_candidate = 0`
- training remains locked
- auto relabel remains locked
- next step: `second_review_or_annotation_guideline_clarification`

Guardrails preserved:

- no training was run
- no automatic relabeling was applied
- original annotations were not modified
- no architecture changes were made
- no loss changes were made

## 2026-06-25 - Phase 9D second review preparation created

Files created:

- `scripts/prepare_phase9d_second_review.py`
- `docs/phase9d_second_review_and_guideline_clarification.md`
- `artifacts/phase9d_second_review_chipped/phase9d_second_review_template.csv`
- `artifacts/phase9d_second_review_chipped/phase9d_annotation_guideline_clarification.md`
- `artifacts/phase9d_second_review_chipped/phase9d_second_review_summary.json`
- `artifacts/phase9d_second_review_chipped/phase9d_second_review_summary.csv`
- `artifacts/phase9d_second_review_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- command prepared Phase 9D second-review artifacts for `error_0064` and `error_0128`
- `status = second_review_template_prepared`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = user_second_review_confirmation_required`

Guardrails preserved:

- no training was run
- no automatic relabeling was applied
- no original annotations were modified
- no architecture changes were made
- no loss changes were made
- no dataset intervention patch was applied

## 2026-06-25 - Phase 9D.5 user second review confirmation

Files created:

- `scripts/confirm_phase9d_second_review.py`
- `artifacts/phase9d_second_review_chipped/phase9d_second_review_template.before_phase9d_5_confirmation.csv`
- `artifacts/phase9d_second_review_chipped/phase9d_second_review_confirmed.csv`
- `artifacts/phase9d_second_review_chipped/phase9d_5_second_review_confirmation_summary.json`
- `artifacts/phase9d_second_review_chipped/phase9d_5_second_review_confirmation_summary.csv`

Files modified:

- `artifacts/phase9d_second_review_chipped/README.md`
- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/confirm_phase9d_second_review.py --phase9d-template artifacts/phase9d_second_review_chipped/phase9d_second_review_template.csv --output artifacts/phase9d_second_review_chipped --focus-class Chipped`

Run summary:

- confirmed second-review decisions for `error_0064` and `error_0128`
- `status = second_review_confirmed_policy_update_required`
- `confirmed_case_count = 2`
- `pending_second_review_count = 0`
- `needs_annotation_policy_update_count = 2`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = phase9e_outcome_summary_and_intervention_scope_definition`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no relabel patch was applied
- no architecture changes were made
- no loss changes were made
- training remained locked

## 2026-06-25 - Phase 9E second review outcome summary and intervention scope definition

Files created:

- `scripts/summarize_phase9e_second_review_outcome.py`
- `docs/phase9e_second_review_outcome_and_intervention_scope.md`
- `artifacts/phase9e_second_review_outcome_chipped/phase9e_case_outcomes.csv`
- `artifacts/phase9e_second_review_outcome_chipped/phase9e_policy_update_cases.csv`
- `artifacts/phase9e_second_review_outcome_chipped/phase9e_intervention_scope_summary.json`
- `artifacts/phase9e_second_review_outcome_chipped/phase9e_intervention_scope_summary.csv`
- `artifacts/phase9e_second_review_outcome_chipped/phase9e_research_interpretation.md`
- `artifacts/phase9e_second_review_outcome_chipped/README.md`

Files modified:

- `docs/experiment_log.md`
- `docs/phase9d_second_review_and_guideline_clarification.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/summarize_phase9e_second_review_outcome.py --phase9d-confirmed artifacts/phase9d_second_review_chipped/phase9d_second_review_confirmed.csv --phase9d-summary artifacts/phase9d_second_review_chipped/phase9d_5_second_review_confirmation_summary.json --phase9c-summary artifacts/phase9c_manual_audit_outcome_chipped/phase9c_manual_audit_outcome_summary.json --output artifacts/phase9e_second_review_outcome_chipped --focus-class Chipped`

Run summary:

- summarized final second-review outcomes for `error_0064` and `error_0128`
- `status = annotation_policy_update_required_before_relabel_or_training`
- `annotation_policy_update_required = true`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `relabel_patch_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = annotation_guideline_revision_before_relabel_or_training`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no relabel patch was created or applied
- no dataset changes were made
- no architecture changes were made
- no loss changes were made
- training remained locked

## 2026-06-28 - Phase 11S local checkpoint publication package

Files modified:

- `scripts/prepare_phase11s_local_checkpoint_publication_package.py`
- `docs/phase11s_local_checkpoint_publication_package.md`
- `docs/experiment_log.md`

Commands run:

- `python -m compileall src scripts`
- `python scripts/prepare_phase11s_local_checkpoint_publication_package.py`

Run summary:

- prepared a local checkpoint publication package bundle from the validated Phase 11R.1 decision
- `status = phase11s_local_checkpoint_publication_package_prepared_metadata_only`
- `phase11r1_validated = true`
- `checkpoint_package_preparation_allowed = true`
- `checkpoint_publication_allowed = false`
- `checkpoint_upload_executed = false`
- `checkpoint_load_executed = false`
- `checkpoint_binary_copied = false`
- `checksum_generated = true`
- `model_card_draft_created = true`
- `release_manifest_created = true`
- `next_allowed_step = phase11t_manual_checkpoint_publication_execution_gate_or_hold`

Guardrails preserved:

- no checkpoint upload was run
- no checkpoint load with model libraries was performed
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation was performed

## 2026-06-27 - Phase 11G.4 real Kaggle mounted YAML path correction preflight package

Files created:

- `scripts/prepare_phase11g_4_real_kaggle_mounted_yaml_path_correction.py`
- `docs/phase11g4_real_kaggle_mounted_yaml_path_correction.md`
- `artifacts/phase11g4_real_kaggle_mounted_yaml_path_correction/phase11g4_real_kaggle_mounted_yaml_path_correction_summary.json`
- `artifacts/phase11g4_real_kaggle_mounted_yaml_path_correction/phase11g4_actual_mounted_kaggle_yaml_snapshot.yaml`
- `artifacts/phase11g4_real_kaggle_mounted_yaml_path_correction/phase11g4_kaggle_yaml_checks.csv`
- `artifacts/phase11g4_real_kaggle_mounted_yaml_path_correction/phase11g4_non_execution_manifest.json`
- `artifacts/phase11g4_real_kaggle_mounted_yaml_path_correction/phase11g4_kaggle_notebook_preflight_commands.md`
- `artifacts/phase11g4_real_kaggle_mounted_yaml_path_correction/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `python -m compileall src scripts`
- `PYTHONPATH=src python scripts/prepare_phase11g_4_real_kaggle_mounted_yaml_path_correction.py`
- `rg -n "/kaggle/input/phase11-staging-dataset-relabel-patch-chipped/staging_dataset_copy" artifacts/phase11g4_real_kaggle_mounted_yaml_path_correction`

Run summary:

- corrected the invalid Phase 11G.3 Kaggle mounted dataset-root assumption to the actual mounted path under `/kaggle/input/datasets/thanhmay2406/...`
- preserved prepare-only and preflight-only execution scope
- kept `ready_for_kaggle_notebook_manual_preflight = true`
- kept `ready_for_training_execution = false`
- kept `training_executed = false`
- kept `evaluation_executed = false`
- kept `inference_executed = false`
- set `next_allowed_step = manual_kaggle_notebook_preflight_with_actual_mounted_yaml_path_before_training_gate`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was mutated
- no staging dataset was mutated
- no Kaggle dataset was uploaded
- no model weights were created
- no checkpoints were created

## 2026-06-27 - Phase 11G.0 Kaggle staging input preflight package preparation

- Scope: prepare-only Kaggle staging-input and preflight package for the approved Phase 11G training path.
- Added `scripts/prepare_phase11g_0_kaggle_staging_input_preflight_package.py` to validate the upstream Phase 11F and Phase 11F.1 gate states, inspect the Phase 9S staging dataset copy, detect the local absolute-path YAML issue, and emit a non-execution Kaggle preflight package.
- Added `docs/phase11g_0_kaggle_staging_input_preflight_package.md` and a new package bundle under `artifacts/phase11g_0_kaggle_staging_input_preflight_package/`.
- Confirmed that Phase 11G must not use the original `data/` dataset tree and must instead use `artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_copy`.
- Kept the phase non-executing and non-mutating: no training, evaluation, or inference ran; the dataset was not mutated; and the package only prepares Kaggle-side preflight materials.
- Next step: upload `staging_dataset_copy` to a Kaggle Dataset and run Kaggle preflight before any real Phase 11G training.

## 2026-06-27 - Phase 11G.1 Kaggle staging dataset upload and slug binding

- Scope: prepare-only Kaggle staging dataset upload and slug-binding package for the approved Phase 11G Kaggle path.
- Added `scripts/prepare_phase11g_1_kaggle_staging_dataset_upload_and_slug_binding.py` to validate the upstream Phase 11G.0 package, re-check the Phase 9S staging dataset inventory, confirm that Kaggle execution does not depend on local `/home/thanhmay` paths, and emit upload/slug-binding templates.
- Added `docs/phase11g_1_kaggle_staging_dataset_upload_and_slug_binding.md` and a new package bundle under `artifacts/phase11g_1_kaggle_staging_dataset_upload_and_slug_binding/`.
- Kept the phase non-executing and non-mutating: no training, evaluation, inference, Kaggle upload, or dataset mutation ran here.
- Next step: provide the real Kaggle Dataset slug and bind the Phase 11G Kaggle YAML for the next Kaggle-side validation phase.

## 2026-06-27 - Phase 11G.2 Kaggle slug-bound preflight handoff validation

- Scope: preflight-only Kaggle slug-bound handoff validation package built on top of Phase 11G.1 slug binding.
- Added `scripts/prepare_phase11g_2_kaggle_slug_bound_preflight_handoff_validation.py` to validate the Phase 11G.1 bound-YAML handoff state, confirm the YAML shape and Kaggle input path semantics, and re-check the approved config lineage remains staging-only and non-executing.
- Added `docs/phase11g_2_kaggle_slug_bound_preflight_handoff_validation.md` and a new package bundle under `artifacts/phase11g_2_kaggle_slug_bound_preflight_handoff_validation/`.
- Kept the phase non-executing and non-mutating: no training, evaluation, inference, Kaggle upload, or dataset mutation ran here.
- Next step: manual Kaggle notebook input preflight remains blocked until Phase 11G.1 is rerun with a real Kaggle Dataset slug and the bound YAML exists.

## 2026-06-27 - Phase 11G.3 real Kaggle slug binding preflight

- Scope: slug-binding and preflight-only rerun after the earlier placeholder Kaggle slug was found in the Phase 11G.1 bound YAML.
- Reused `scripts/prepare_phase11g_1_kaggle_staging_dataset_upload_and_slug_binding.py` with `--kaggle-dataset-slug thanhmay2406/phase11-staging-dataset-relabel-patch-chipped` to regenerate the Phase 11G.1 bound YAML with the real Kaggle dataset id.
- Added `scripts/prepare_phase11g_3_real_kaggle_slug_binding_preflight.py` to validate the real-slug YAML path, split entries, `nc`, class-name order, and strict placeholder removal without running any execution step.
- Added `docs/phase11g3_real_kaggle_slug_binding_preflight.md` and a new package bundle under `artifacts/phase11g3_real_kaggle_slug_binding_preflight/`.
- Kept the phase non-executing and non-mutating: no training, evaluation, inference, Kaggle upload, weights, checkpoints, or dataset mutation ran here.
- Next step: manual Kaggle Notebook preflight with the real slug before any later training gate.

## 2026-06-26 - Phase 10A repository hygiene audit tooling and cleanup planning

Files created:

- `scripts/audit_phase10a_repo_hygiene.py`
- `docs/phase10a_repo_hygiene_and_cleanup_plan.md`
- `artifacts/phase10a_repo_hygiene/phase10a_repo_hygiene_summary.json`
- `artifacts/phase10a_repo_hygiene/phase10a_large_directories.csv`
- `artifacts/phase10a_repo_hygiene/phase10a_large_files.csv`
- `artifacts/phase10a_repo_hygiene/phase10a_cleanup_candidates.csv`
- `artifacts/phase10a_repo_hygiene/phase10a_gitignore_recommendations.txt`
- `artifacts/phase10a_repo_hygiene/phase10a_retention_plan.md`
- `artifacts/phase10a_repo_hygiene/phase10a_non_mutation_manifest.json`
- `artifacts/phase10a_repo_hygiene/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- created Phase 10A audit-only repository hygiene tooling and reports
- no cleanup was applied
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- outputs were written under `artifacts/phase10a_repo_hygiene/`
- next step is manual review of cleanup candidates before `phase10b_apply_safe_cleanup_after_manual_review`

## 2026-06-26 - Phase 10B safe repository cleanup tooling

Files created:

- `scripts/apply_phase10b_safe_repo_cleanup.py`
- `docs/phase10b_safe_repo_cleanup.md`
- `artifacts/phase10b_safe_cleanup/phase10b_safe_cleanup_summary.json`
- `artifacts/phase10b_safe_cleanup/phase10b_planned_operations.csv`
- `artifacts/phase10b_safe_cleanup/phase10b_completed_operations.csv`
- `artifacts/phase10b_safe_cleanup/phase10b_failed_operations.csv`
- `artifacts/phase10b_safe_cleanup/phase10b_archived_files_manifest.csv`
- `artifacts/phase10b_safe_cleanup/phase10b_deleted_cache_manifest.csv`
- `artifacts/phase10b_safe_cleanup/phase10b_non_mutation_manifest.json`
- `artifacts/phase10b_safe_cleanup/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10B safe cleanup script with default dry-run behavior
- default run was dry-run before an explicit apply run
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10b_safe_cleanup/`

## 2026-06-26 - Phase 10C safe gitignore hygiene tooling

Files created:

- `scripts/apply_phase10c_safe_gitignore_hygiene.py`
- `docs/phase10c_safe_gitignore_hygiene.md`
- `artifacts/phase10c_safe_gitignore_hygiene/phase10c_gitignore_hygiene_summary.json`
- `artifacts/phase10c_safe_gitignore_hygiene/phase10c_existing_gitignore_snapshot.txt`
- `artifacts/phase10c_safe_gitignore_hygiene/phase10c_proposed_gitignore_patterns.csv`
- `artifacts/phase10c_safe_gitignore_hygiene/phase10c_applied_gitignore_patterns.csv`
- `artifacts/phase10c_safe_gitignore_hygiene/phase10c_skipped_patterns.csv`
- `artifacts/phase10c_safe_gitignore_hygiene/phase10c_non_mutation_manifest.json`
- `artifacts/phase10c_safe_gitignore_hygiene/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10C safe gitignore hygiene script with default dry-run behavior
- default run was dry-run before an explicit apply run
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10c_safe_gitignore_hygiene/`

## 2026-06-26 - Phase 10D clean repo audit checkpoint tooling

Files created:

- `scripts/audit_phase10d_clean_repo_checkpoint.py`
- `docs/phase10d_clean_repo_checkpoint.md`
- `artifacts/phase10d_clean_repo_checkpoint/phase10d_clean_repo_checkpoint_summary.json`
- `artifacts/phase10d_clean_repo_checkpoint/phase10d_git_status_categorized.csv`
- `artifacts/phase10d_clean_repo_checkpoint/phase10d_ignored_status_snapshot.csv`
- `artifacts/phase10d_clean_repo_checkpoint/phase10d_large_path_report.csv`
- `artifacts/phase10d_clean_repo_checkpoint/phase10d_repo_hygiene_recommendations.md`
- `artifacts/phase10d_clean_repo_checkpoint/phase10d_non_mutation_manifest.json`
- `artifacts/phase10d_clean_repo_checkpoint/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10D clean repo checkpoint audit script
- Phase 10D was audit-only and did not move, delete, or archive files
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10d_clean_repo_checkpoint/`

## 2026-06-26 - Phase 10E safe deletion candidate audit tooling

Files created:

- `scripts/audit_phase10e_safe_deletion_candidates.py`
- `docs/phase10e_safe_deletion_candidates.md`
- `artifacts/phase10e_safe_deletion_candidates/phase10e_safe_deletion_summary.json`
- `artifacts/phase10e_safe_deletion_candidates/phase10e_deletion_candidates.csv`
- `artifacts/phase10e_safe_deletion_candidates/phase10e_keep_protected_report.csv`
- `artifacts/phase10e_safe_deletion_candidates/phase10e_manual_review_before_delete.csv`
- `artifacts/phase10e_safe_deletion_candidates/phase10e_environment_optional_delete_report.csv`
- `artifacts/phase10e_safe_deletion_candidates/phase10e_recommended_next_deletion_commands.sh`
- `artifacts/phase10e_safe_deletion_candidates/phase10e_non_mutation_manifest.json`
- `artifacts/phase10e_safe_deletion_candidates/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10E safe deletion candidate audit script
- Phase 10E was audit-only and did not delete, move, archive, stage, or commit files
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10e_safe_deletion_candidates/`

## 2026-06-26 - Phase 10F approved safe deletions tooling

Files created:

- `scripts/apply_phase10f_approved_safe_deletions.py`
- `docs/phase10f_approved_safe_deletions.md`
- `artifacts/phase10f_approved_safe_deletions/phase10f_safe_deletion_summary.json`
- `artifacts/phase10f_approved_safe_deletions/phase10f_planned_deletions.csv`
- `artifacts/phase10f_approved_safe_deletions/phase10f_completed_deletions.csv`
- `artifacts/phase10f_approved_safe_deletions/phase10f_failed_deletions.csv`
- `artifacts/phase10f_approved_safe_deletions/phase10f_skipped_deletions.csv`
- `artifacts/phase10f_approved_safe_deletions/phase10f_protected_refusal_report.csv`
- `artifacts/phase10f_approved_safe_deletions/phase10f_non_mutation_manifest.json`
- `artifacts/phase10f_approved_safe_deletions/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10F approved safe deletions script with default dry-run behavior
- default run was dry-run before an explicit apply run
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10f_approved_safe_deletions/`

## 2026-06-26 - Phase 10G post-deletion repo hygiene verification tooling

Files created:

- `scripts/audit_phase10g_post_deletion_repo_hygiene.py`
- `docs/phase10g_post_deletion_repo_hygiene.md`
- `artifacts/phase10g_post_deletion_repo_hygiene/phase10g_post_deletion_hygiene_summary.json`
- `artifacts/phase10g_post_deletion_repo_hygiene/phase10g_git_status_categorized.csv`
- `artifacts/phase10g_post_deletion_repo_hygiene/phase10g_phase10f_deletion_verification.csv`
- `artifacts/phase10g_post_deletion_repo_hygiene/phase10g_large_path_report.csv`
- `artifacts/phase10g_post_deletion_repo_hygiene/phase10g_remaining_dirty_state_recommendations.md`
- `artifacts/phase10g_post_deletion_repo_hygiene/phase10g_non_mutation_manifest.json`
- `artifacts/phase10g_post_deletion_repo_hygiene/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10G post-deletion repo hygiene verification audit script
- Phase 10G was audit-only and did not delete, move, archive, stage, or commit files
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10g_post_deletion_repo_hygiene/`

## 2026-06-26 - Phase 10H remaining package review tooling

Files created:

- `scripts/audit_phase10h_remaining_package_review.py`
- `docs/phase10h_remaining_package_review.md`
- `artifacts/phase10h_remaining_package_review/phase10h_remaining_package_review_summary.json`
- `artifacts/phase10h_remaining_package_review/phase10h_package_review_candidates.csv`
- `artifacts/phase10h_remaining_package_review/phase10h_safe_delete_after_review.csv`
- `artifacts/phase10h_remaining_package_review/phase10h_manual_review_before_delete.csv`
- `artifacts/phase10h_remaining_package_review/phase10h_keep_protected_report.csv`
- `artifacts/phase10h_remaining_package_review/phase10h_recommended_next_deletion_commands.sh`
- `artifacts/phase10h_remaining_package_review/phase10h_non_mutation_manifest.json`
- `artifacts/phase10h_remaining_package_review/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10H remaining package review audit script
- Phase 10H was audit-only and did not delete, move, archive, stage, or commit files
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10h_remaining_package_review/`

## 2026-06-26 - Phase 10I reviewed package deletions tooling

Files created:

- `scripts/apply_phase10i_reviewed_package_deletions.py`
- `docs/phase10i_reviewed_package_deletions.md`
- `artifacts/phase10i_reviewed_package_deletions/phase10i_reviewed_package_deletion_summary.json`
- `artifacts/phase10i_reviewed_package_deletions/phase10i_planned_deletions.csv`
- `artifacts/phase10i_reviewed_package_deletions/phase10i_completed_deletions.csv`
- `artifacts/phase10i_reviewed_package_deletions/phase10i_failed_deletions.csv`
- `artifacts/phase10i_reviewed_package_deletions/phase10i_skipped_deletions.csv`
- `artifacts/phase10i_reviewed_package_deletions/phase10i_protected_refusal_report.csv`
- `artifacts/phase10i_reviewed_package_deletions/phase10i_non_mutation_manifest.json`
- `artifacts/phase10i_reviewed_package_deletions/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10I reviewed package deletions script with default dry-run behavior
- default run was dry-run before an explicit apply run
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10i_reviewed_package_deletions/`

## 2026-06-26 - Phase 10J final cleanup commit readiness tooling

Files created:

- `scripts/audit_phase10j_final_cleanup_commit_readiness.py`
- `docs/phase10j_final_cleanup_commit_readiness.md`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_final_cleanup_commit_readiness_summary.json`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_git_status_commit_categories.csv`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_phase10i_deletion_verification.csv`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_remaining_package_review.csv`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_large_path_report.csv`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_recommended_commit_plan.md`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_recommended_git_add_commands.sh`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_do_not_stage_report.csv`
- `artifacts/phase10j_final_cleanup_commit_readiness/phase10j_non_mutation_manifest.json`
- `artifacts/phase10j_final_cleanup_commit_readiness/README.md`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 10J final cleanup commit-readiness audit script
- Phase 10J was audit-only and did not delete, move, archive, stage, or commit files
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- Phase 9Z.5 approval state was preserved
- outputs were written under `artifacts/phase10j_final_cleanup_commit_readiness/`

## 2026-06-26 - Phase 11A research resume context package

Files created:

- `scripts/prepare_phase11a_research_resume_context.py`
- `docs/phase11a_research_resume_context.md`
- `artifacts/phase11a_research_resume_context/phase11a_research_resume_summary.json`
- `artifacts/phase11a_research_resume_context/phase11a_collected_files_manifest.csv`
- `artifacts/phase11a_research_resume_context/phase11a_skipped_files_manifest.csv`
- `artifacts/phase11a_research_resume_context/phase11a_missing_expected_files.csv`
- `artifacts/phase11a_research_resume_context/phase11a_research_state_brief.md`
- `artifacts/phase11a_research_resume_context/phase11a_non_mutation_manifest.json`
- `artifacts/phase11a_research_resume_context/README.md`
- `artifacts/phase11a_research_resume_context/phase11a_research_resume_context.zip`

Files modified:

- `docs/experiment_log.md`

Run summary:

- added the Phase 11A collection-only research resume context script
- Phase 11A validates the Phase 10B through Phase 10I cleanup summary chain before packaging context
- Phase 11A collects lightweight docs, configs, summaries, manifests, reports, and baseline metadata while skipping heavy visuals and archive-like files by rule
- Phase 11A writes a portable zip package under `artifacts/phase11a_research_resume_context/`
- no training was run
- no evaluation was run
- no inference was run
- no dataset mutation occurred
- no labels were mutated
- no approval state was mutated
- Phase 9Z.5 approval state was preserved

## 2026-06-26 - Phase 9Y approved staging training execution guard for Chipped

Files created:

- `scripts/execute_phase9y_approved_staging_training.py`
- `docs/phase9y_approved_staging_training_execution_chipped.md`
- `artifacts/phase9y_approved_staging_training_chipped/phase9y_training_execution_summary.json`
- `artifacts/phase9y_approved_staging_training_chipped/phase9y_training_command.txt`
- `artifacts/phase9y_approved_staging_training_chipped/phase9y_preflight_checks.csv`
- `artifacts/phase9y_approved_staging_training_chipped/phase9y_training_run_manifest.json`
- `artifacts/phase9y_approved_staging_training_chipped/phase9y_non_mutation_manifest.json`
- `artifacts/phase9y_approved_staging_training_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py --execute-approved-training`

Run summary:

- implemented a strict Phase 9Y executor that revalidates Phase 9V, Phase 9W, Phase 9X, the Phase 9S staging yaml target, and non-mutation scope before any training may run
- default mode remains blocked unless `--execute-approved-training` is provided
- actual execution still remained blocked in verification because the current Phase 9X approval provenance matched the bundled example approval snapshot rather than a clearly real human approval record
- `status = phase9y_blocked_invalid_phase9v_phase9w_phase9x_or_training_scope`
- `approved_for_training_execution = false`
- `approval_provenance_is_real_human_approval = false`
- `training_command_executed = false`
- `training_executed = false`
- `evaluation_executed = false`
- `inference_executed = false`
- `original_dataset_mutated = false`
- `staging_dataset_mutated_by_phase9y = false`
- `phase9s_staging_dataset_labels_modified_by_phase9y = false`
- `next_allowed_step = provide_real_human_phase9x_approval_and_rerun_phase9y_with_execute_flag`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no Phase 9S staging dataset labels were modified
- no dataset split was changed
- no model architecture was changed
- no loss configuration was changed
- the reviewed Phase 9V config lineage remained intact

## 2026-06-26 - Phase 9Y.1 approval provenance hardening before real training execution

Files modified:

- `scripts/execute_phase9y_approved_staging_training.py`
- `docs/phase9y_approved_staging_training_execution_chipped.md`
- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py --execute-approved-training`

Run summary:

- narrowed Phase 9Y to classify approval provenance explicitly as `approval_missing`, `approval_example_fixture`, `approval_real_human_valid`, or `approval_real_human_invalid`
- added the dedicated approval-only blocked status `phase9y_blocked_missing_real_human_training_approval`
- kept the broader invalid status reserved for Phase 9V, Phase 9W, Phase 9X, dataset target, lineage, or scope failures
- added explicit summary fields for `approval_provenance_class`, `approval_is_example_fixture`, `real_human_approval_detected`, `real_human_approval_required`, `training_execution_block_reason`, and `safe_to_execute_training`
- added explicit Phase 9Y.1 audit rows for `phase9v_config_valid`, `phase9w_command_review_valid`, `phase9x_summary_present`, `phase9x_approval_scope_valid`, `approval_not_example_fixture`, `real_human_approval_present`, `staging_dataset_target_confirmed`, and `original_dataset_not_targeted`
- both verification runs remained blocked because the current approval provenance still resolves to the bundled Phase 9X example fixture rather than a real human approval
- `status = phase9y_blocked_missing_real_human_training_approval`
- `approval_provenance_class = approval_example_fixture`
- `approval_is_example_fixture = true`
- `real_human_approval_detected = false`
- `safe_to_execute_training = false`
- `training_command_executed = false`
- `training_executed = false`
- `evaluation_executed = false`
- `inference_executed = false`
- `next_allowed_step = provide_real_human_phase9x_approval_and_rerun_phase9y_with_execute_flag`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no Phase 9S staging dataset labels were modified
- no Phase 9V config values were changed
- no model architecture was changed
- no loss configuration was changed
- no training settings were changed

## 2026-06-26 - Phase 9Z Kaggle handoff package for approved Phase 9Y staging training execution

Files created:

- `scripts/prepare_phase9z_kaggle_training_handoff.py`
- `docs/phase9z_kaggle_training_handoff_chipped.md`
- `artifacts/phase9z_kaggle_training_handoff_chipped/phase9z_kaggle_handoff_summary.json`
- `artifacts/phase9z_kaggle_training_handoff_chipped/phase9z_kaggle_file_manifest.csv`
- `artifacts/phase9z_kaggle_training_handoff_chipped/phase9z_kaggle_preflight_checks.csv`
- `artifacts/phase9z_kaggle_training_handoff_chipped/phase9z_kaggle_run_commands.sh`
- `artifacts/phase9z_kaggle_training_handoff_chipped/phase9z_real_human_approval_required.md`
- `artifacts/phase9z_kaggle_training_handoff_chipped/phase9z_real_human_approval_template.csv`
- `artifacts/phase9z_kaggle_training_handoff_chipped/phase9z_upload_manifest.json`
- `artifacts/phase9z_kaggle_training_handoff_chipped/phase9z_non_execution_manifest.json`
- `artifacts/phase9z_kaggle_training_handoff_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/prepare_phase9z_kaggle_training_handoff.py`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py`

Run summary:

- prepared a Kaggle handoff package from the current Phase 9V, 9W, 9X, and 9Y artifact chain without executing training
- verified the Phase 9V config still points to the Phase 9S staging dataset yaml and does not target the original dataset
- verified the Phase 9W command review summary, Phase 9X approval summary, and Phase 9Y summary all exist and remain consistent
- verified the current Phase 9Y state remains intentionally blocked because the approval provenance is still `approval_example_fixture`
- created a reviewer-editable approval template that remains pending by default and preserves the Phase 9X-required approval columns
- created a commented Kaggle command sequence that stops short of any automatic training execution
- documented the required next step as manual real human approval followed by Phase 9X rerun, Phase 9Y preflight rerun, and Kaggle training only if `safe_to_execute_training=true`
- `status = phase9z_kaggle_handoff_prepared`
- `phase9y_status = phase9y_blocked_missing_real_human_training_approval`
- `phase9y_approval_provenance_class = approval_example_fixture`
- `phase9y_safe_to_execute_training = false`
- `next_allowed_step = manual_real_human_approval_then_phase9x_phase9y_kaggle_execution`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no Phase 9S staging dataset was modified
- no Phase 9V, Phase 9W, Phase 9X, or Phase 9Y artifacts were modified except by being read
- no model architecture was changed
- no loss configuration was changed
- no dataset split was changed

## 2026-06-26 - Phase 9Z.1 Kaggle runtime dataset assembly audit for Phase 9Y staging training

Files created:

- `scripts/prepare_phase9z1_kaggle_runtime_dataset_assembly.py`
- `docs/phase9z1_kaggle_runtime_dataset_assembly_chipped.md`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_runtime_assembly_summary.json`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_original_image_counts.csv`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_staging_label_counts.csv`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_image_label_match_check.csv`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_missing_image_matches.csv`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_extra_original_images.csv`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_kaggle_runtime_dataset_yaml.yaml`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_phase9v_kaggle_runtime_adapter.yaml`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_kaggle_runtime_commands.sh`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/phase9z1_non_execution_manifest.json`
- `artifacts/phase9z1_kaggle_runtime_dataset_assembly_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/prepare_phase9z1_kaggle_runtime_dataset_assembly.py`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py`

Run summary:

- created a runtime assembly audit package showing that Phase 9S is labels-only and must be paired with original images at Kaggle runtime
- verified `phase9s_image_count_total = 0`
- verified `phase9s_label_count_total = 6041`
- verified `original_image_count_total = 6041`
- verified exact split-preserving stem matches across all splits:
- `train: 4219 labels, 4219 images, 0 missing matches, 0 extra original images`
- `valid: 916 labels, 916 images, 0 missing matches, 0 extra original images`
- `test: 906 labels, 906 images, 0 missing matches, 0 extra original images`
- created a Kaggle runtime YAML copy pointing to `/kaggle/working/phase9s_runtime_dataset`
- created a non-authoritative Phase 9V Kaggle runtime adapter copy without overwriting the original Phase 9V config artifact
- kept default mode as audit-only with no-copy and no assembly unless `--assemble` is supplied
- `status = phase9z1_runtime_dataset_audit_prepared`
- `phase9y_status = phase9y_blocked_missing_real_human_training_approval`
- `phase9y_approval_provenance_class = approval_example_fixture`
- `phase9y_safe_to_execute_training = false`
- `next_allowed_step = manual_real_human_approval_then_phase9x_phase9y_kaggle_execution`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no Phase 9S staging labels were modified
- no Phase 9V, Phase 9W, Phase 9X, Phase 9Y, or Phase 9Z artifacts were modified except by being read
- no model architecture was changed
- no loss configuration was changed
- no dataset split was changed
- no fake or auto-filled real human approval was created

## 2026-06-26 - Phase 9Z.2 final Kaggle execution bundle preparation for Phase 9Y staging training

Files created:

- `scripts/prepare_phase9z2_kaggle_execution_bundle.py`
- `docs/phase9z2_kaggle_execution_bundle_chipped.md`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/phase9z2_execution_bundle_summary.json`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/phase9z2_execution_bundle_manifest.csv`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/phase9z2_excluded_paths_report.csv`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/phase9z2_kaggle_input_requirements.md`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/phase9z2_kaggle_runtime_runbook.md`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/phase9z2_kaggle_commands.sh`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/phase9z2_real_human_approval_template.csv`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/phase9z2_non_execution_manifest.json`
- `artifacts/phase9z2_kaggle_execution_bundle_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/prepare_phase9z2_kaggle_execution_bundle.py`
- `.venv/bin/python scripts/prepare_phase9z1_kaggle_runtime_dataset_assembly.py`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py`

Run summary:

- prepared a final Kaggle execution bundle and runbook for Phase 9Y staging training without executing any training
- verified the bundle manifest includes the required `src/`, execution scripts, configs, staging-label artifacts, approval-gate artifacts, and Phase 9Z and Phase 9Z.1 handoff materials
- verified original dataset images remain excluded by default and are documented as a separate read-only Kaggle input unless explicitly requested
- verified the excluded-path report covers `.venv`, `.git`, `__pycache__`, cache folders, `runs`, `experiments`, heavy checkpoint patterns, and Kaggle credentials
- verified the generated Kaggle command file is fully commented and does not auto-execute training
- included both `.venv/bin/python` and plain `python` command styles for Kaggle environments that do not use `.venv`
- `status = phase9z2_kaggle_execution_bundle_prepared`
- `include_original_dataset_images = false`
- `included_manifest_path_count = 18`
- `excluded_report_count = 19`
- `phase9y_status = phase9y_blocked_missing_real_human_training_approval`
- `phase9y_approval_provenance_class = approval_example_fixture`
- `phase9y_safe_to_execute_training = false`
- `next_allowed_step = manual_real_human_approval_then_phase9x_phase9y_kaggle_execution`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no Phase 9S staging labels were modified
- no Phase 9V original config was overwritten
- no Phase 9S original YAML was overwritten
- no fake or auto-filled real human approval was created
- all training commands remain explicitly gated by real human approval and a safe Phase 9Y preflight state

## 2026-06-26 - Phase 9Z.3 runnable Kaggle archive and runtime path smoke-check for Phase 9Y staging training

Files created:

- `scripts/prepare_phase9z3_runnable_kaggle_archive.py`
- `docs/phase9z3_runnable_kaggle_archive_chipped.md`
- `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_runnable_kaggle_archive.zip`
- `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_runnable_archive_summary.json`
- `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_archive_manifest.csv`
- `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_archive_exclusions.csv`
- `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_kaggle_commands.sh`
- `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_kaggle_runtime_path_smoke_check.md`
- `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_non_execution_manifest.json`
- `artifacts/phase9z3_runnable_kaggle_archive_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/prepare_phase9z3_runnable_kaggle_archive.py`
- `unzip -l artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_runnable_kaggle_archive.zip | sed -n '1,120p'`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py`

Run summary:

- created a real runnable Kaggle zip archive containing the minimal project structure for runtime dataset assembly, approval rerun, and Phase 9Y preflight
- verified the archive includes `src/`, `scripts/`, `configs/`, the required Phase 9 docs, and the Phase 9S, 9V, 9W, 9X, 9Y, 9Z, 9Z.1, and 9Z.2 artifact directories
- verified original dataset images remain excluded by default, including the `Phase 9S` staging-copy image subtree, so the archive stays labels-only for runtime assembly purposes
- verified the archive exclusion manifest covers `.venv`, `.git`, `__pycache__`, cache folders, `runs`, `experiments`, checkpoint patterns, Kaggle credentials, and default original-image exclusions
- verified the generated Kaggle command file remains fully commented and includes explicit `--phase9s-label-root`, `--runtime-output-root`, and Kaggle original-image placeholder arguments
- verified the runtime path smoke-check document explains how to validate extracted paths, Kaggle input mounts, writable runtime output paths, and the no-write rule for `/kaggle/input`
- verified Phase 9Y still does not expose a runtime adapter or config override argument for the separate Phase 9Z.1 Kaggle runtime adapter copy
- `status = phase9z3_runnable_kaggle_archive_prepared`
- `archive_size_bytes = 3177117`
- `archive_file_count = 6216`
- `phase9y_supports_runtime_adapter = false`
- `ready_for_kaggle_upload = true`
- `ready_for_kaggle_training_execution = false`
- `phase9y_status = phase9y_blocked_missing_real_human_training_approval`
- `phase9y_approval_provenance_class = approval_example_fixture`
- `phase9y_safe_to_execute_training = false`
- `next_allowed_step = narrow_phase9y_runtime_adapter_argument_patch_then_manual_real_human_approval_then_phase9x_phase9y_kaggle_execution`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no Phase 9S staging labels were modified
- no original dataset images were bundled by default
- no Phase 9V original config was overwritten
- no Phase 9S original YAML was overwritten
- no fake or auto-filled real human approval was created
- the runnable archive does not claim Kaggle training execution readiness while the Phase 9Y runtime-adapter gap remains unresolved

## 2026-06-26 - Phase 9Z.4 Phase 9Y runtime adapter argument patch for Chipped

Files created:

- `docs/phase9z4_phase9y_runtime_adapter_patch_chipped.md`
- `artifacts/phase9z4_phase9y_runtime_adapter_patch_chipped/phase9z4_runtime_adapter_summary.json`
- `artifacts/phase9z4_phase9y_runtime_adapter_patch_chipped/phase9z4_runtime_adapter_checks.csv`
- `artifacts/phase9z4_phase9y_runtime_adapter_patch_chipped/phase9z4_non_execution_manifest.json`
- `artifacts/phase9z4_phase9y_runtime_adapter_patch_chipped/README.md`

Files modified:

- `scripts/execute_phase9y_approved_staging_training.py`
- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py --allow-runtime-adapter-check --phase9z3-archive-summary artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_runnable_archive_summary.json --runtime-root /kaggle/working/phase9z3_runtime --kaggle-input-root /kaggle/input/phase9z3-runnable-kaggle-archive`

Run summary:

- added narrow runtime adapter CLI support to Phase 9Y through `--runtime-root`, `--kaggle-input-root`, `--phase9z3-archive-summary`, and `--allow-runtime-adapter-check`
- preserved the default Phase 9Y behavior so the normal run still returned `status = phase9y_blocked_missing_real_human_training_approval`
- verified the Phase 9Z.3 archive summary was readable and still reported `ready_for_kaggle_upload = true`
- verified the real Phase 9Z.3 zip archive existed at `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_runnable_kaggle_archive.zip`
- verified the supplied Kaggle runtime root and Kaggle input root parsed as valid absolute paths
- verified the existing Phase 9Z.1 runtime adapter still pointed to `phase9z1_kaggle_runtime_dataset_yaml.yaml`
- verified the runtime layout can be derived as a labels-only archive plus an external original-image source
- verified the Phase 9Z.4 adapter-check run returned `status = phase9y_runtime_adapter_patch_prepared`
- `phase = 9Z.4`
- `phase9y_supports_runtime_adapter = true`
- `ready_for_kaggle_upload = true`
- `ready_for_kaggle_training_execution = false`
- `training_allowed = false`
- `training_executed = false`
- `evaluation_executed = false`
- `inference_executed = false`
- `dataset_mutated = false`
- `original_dataset_mutated = false`
- `staging_dataset_mutated = false`
- `approval_source_required = real_human_approval`
- `current_approval_source = approval_example_fixture`
- `next_allowed_step = rerun_phase9y_with_real_human_approval_against_phase9z3_runtime_archive`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no staging dataset was modified
- no original images were copied into the Phase 9Z.3 archive
- no labels were changed
- no model architecture changes were made
- no loss changes were made
- no training config was changed in a way that executes real training
- no fixture approval was upgraded into real approval
- no training approval was auto-granted

## 2026-06-26 - Phase 9Z.5 real-human approval rerun preparation for Phase 9Y runtime archive

Files created:

- `scripts/prepare_phase9z5_real_human_approval_rerun.py`
- `docs/phase9z5_real_human_approval_rerun_chipped.md`
- `artifacts/phase9z5_real_human_approval_rerun_chipped/phase9z5_real_human_approval_rerun_summary.json`
- `artifacts/phase9z5_real_human_approval_rerun_chipped/phase9z5_real_human_approval_checks.csv`
- `artifacts/phase9z5_real_human_approval_rerun_chipped/phase9z5_real_human_approval_template.csv`
- `artifacts/phase9z5_real_human_approval_rerun_chipped/phase9z5_real_human_approval_checklist.md`
- `artifacts/phase9z5_real_human_approval_rerun_chipped/phase9z5_non_execution_manifest.json`
- `artifacts/phase9z5_real_human_approval_rerun_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/prepare_phase9z5_real_human_approval_rerun.py`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py`
- `.venv/bin/python scripts/execute_phase9y_approved_staging_training.py --allow-runtime-adapter-check --phase9z3-archive-summary artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_runnable_archive_summary.json --runtime-root /kaggle/working/phase9z3_runtime --kaggle-input-root /kaggle/input/phase9z3-runnable-kaggle-archive`

Run summary:

- created a dedicated Phase 9Z.5 approval-preparation script for future Phase 9Y reruns against the real Phase 9Z.3 runtime archive
- created a human-editable approval template and checklist for the runtime archive rerun path
- verified the live Phase 9Z.3 summary remained upload-ready
- verified the live Phase 9Z.4 summary still reported `phase9y_supports_runtime_adapter = true`
- verified the Phase 9Z.4 path remained non-executing with `training_executed = false`
- verified the runtime archive zip still existed at `artifacts/phase9z3_runnable_kaggle_archive_chipped/phase9z3_runnable_kaggle_archive.zip`
- verified the Phase 9Y script still exposed `--approval-csv`, `--runtime-root`, `--kaggle-input-root`, and `--phase9z3-archive-summary`
- default Phase 9Z.5 run produced `status = phase9z5_waiting_for_real_human_approval`
- default Phase 9Z.5 run produced `current_approval_source = missing`
- default Phase 9Z.5 run produced `approval_validated = false`
- default Phase 9Z.5 run produced `ready_for_kaggle_training_execution = false`
- default Phase 9Z.5 run produced `training_allowed = false`
- default Phase 9Y run still produced `status = phase9y_blocked_missing_real_human_training_approval`
- Phase 9Y runtime adapter check still produced `status = phase9y_runtime_adapter_patch_prepared`
- `next_allowed_step = collect_real_human_approval_for_phase9y_runtime_archive_rerun`

Current gate status:

- no real-human approval file was provided in this Phase 9Z.5 run
- the bundled example fixture still does not qualify as real approval
- the repo is prepared for a future rerun validation step once a genuine human approval CSV is supplied

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no staging dataset was modified
- no original images were copied into the archive
- no model architecture changes were made
- no loss changes were made
- no training config was changed toward real execution
- no real-human approval was fabricated
- no fixture approval was upgraded into real approval
- no training approval was auto-granted

## 2026-06-25 - Phase 9W final command review before staging training for Chipped

Files created:

- `scripts/review_phase9w_final_training_command.py`
- `docs/phase9w_final_command_review_before_training_chipped.md`
- `artifacts/phase9w_final_command_review_chipped/phase9w_final_command_review_summary.json`
- `artifacts/phase9w_final_command_review_chipped/phase9w_training_command_review.csv`
- `artifacts/phase9w_final_command_review_chipped/phase9w_training_execution_approval_template.csv`
- `artifacts/phase9w_final_command_review_chipped/phase9w_non_execution_manifest.json`
- `artifacts/phase9w_final_command_review_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/review_phase9w_final_training_command.py`

Run summary:

- reviewed the Phase 9V staging training config and command preview without executing them
- `status = phase9w_final_command_review_prepared`
- `training_command_reviewed = true`
- `training_command_executed = false`
- `training_executed = false`
- `evaluation_executed = false`
- `inference_executed = false`
- `original_dataset_mutated = false`
- `staging_dataset_mutated_by_phase9w = false`
- `approved_for_training_execution_in_phase9w = false`
- `next_allowed_step = collect_human_training_execution_approval_or_phase9x_execute_approved_staging_training`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset files were modified
- no staging dataset files were modified
- no new relabel patch was applied
- XAI remained support signal only and was not used as ground truth

## 2026-06-25 - Phase 9V staging training config preparation only for Chipped

Files created:

- `scripts/prepare_phase9v_staging_training_config.py`
- `configs/train/phase9v_staging_chipped_yolov8n.yaml`
- `docs/phase9v_staging_training_config_chipped.md`
- `artifacts/phase9v_staging_training_config_chipped/phase9v_training_config_summary.json`
- `artifacts/phase9v_staging_training_config_chipped/phase9v_training_config_summary.csv`
- `artifacts/phase9v_staging_training_config_chipped/phase9v_training_config_integrity_check.csv`
- `artifacts/phase9v_staging_training_config_chipped/phase9v_training_command_preview.sh`
- `artifacts/phase9v_staging_training_config_chipped/phase9v_training_command_preview.txt`
- `artifacts/phase9v_staging_training_config_chipped/phase9v_config_review_checklist.md`
- `artifacts/phase9v_staging_training_config_chipped/phase9v_non_mutation_manifest.json`
- `artifacts/phase9v_staging_training_config_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/prepare_phase9v_staging_training_config.py`

Run summary:

- consumed the approved Phase 9U training gate
- created a staging-only training config for later review
- `status = phase9v_staging_training_config_prepared`
- `training_config_prepared = true`
- `training_config_points_to_staging_dataset = true`
- `training_config_points_to_original_dataset = false`
- `training_command_preview_created = true`
- `training_command_executed = false`
- `training_executed = false`
- `evaluation_executed = false`
- `inference_executed = false`
- `next_allowed_step = phase9w_final_command_review_before_staging_training_execution`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset files were modified
- no staging dataset files were modified
- no new relabel patch was applied
- XAI remained support signal only and was not used as ground truth

## 2026-06-25 - Phase 9U human training gate decision for staging dataset copy for Chipped

Files created:

- `scripts/decide_phase9u_human_training_gate_staging_dataset_copy.py`
- `docs/phase9u_human_training_gate_staging_dataset_copy_chipped.md`
- `artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/phase9u_training_gate_decision_summary.json`
- `artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/phase9u_training_gate_decision_summary.csv`
- `artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/phase9u_human_training_gate_decision_template.csv`
- `artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/phase9u_human_training_gate_decision_filled.csv`
- `artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/phase9u_human_training_gate_decision_used.csv`
- `artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/phase9u_gate_checklist.md`
- `artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/phase9u_non_mutation_manifest.json`
- `artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/decide_phase9u_human_training_gate_staging_dataset_copy.py`
- `.venv/bin/python scripts/decide_phase9u_human_training_gate_staging_dataset_copy.py --human-decision-csv artifacts/phase9u_human_training_gate_staging_dataset_copy_chipped/phase9u_human_training_gate_decision_filled.csv`

Run summary:

- default run without human decision stayed blocked with `status = phase9u_blocked_waiting_human_training_gate_decision`
- blocked run kept `training_allowed = false`
- blocked run kept `approved_for_training = false`
- approved test run produced `status = phase9u_staging_dataset_training_gate_approved`
- approved test run set `training_allowed = true`
- approved test run set `approved_for_training = true`
- approved test run set `approved_for_training_config_preparation = true`
- approved test run kept `training_executed = false`
- approved test run kept `evaluation_executed = false`
- approved test run kept `inference_executed = false`
- approved test run set `next_allowed_step = phase9v_prepare_staging_training_config_no_training_yet`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset files were modified
- no staging labels were modified
- no new relabel patch was applied
- XAI remained support signal only and was not used as ground truth

## 2026-06-25 - Phase 9T manual review package for staging dataset copy before training gate for Chipped

Files created:

- `scripts/prepare_phase9t_manual_review_staging_dataset_copy.py`
- `docs/phase9t_manual_review_staging_dataset_copy_chipped.md`
- `artifacts/phase9t_manual_review_staging_dataset_copy_chipped/phase9t_manual_review_summary.json`
- `artifacts/phase9t_manual_review_staging_dataset_copy_chipped/phase9t_manual_review_summary.csv`
- `artifacts/phase9t_manual_review_staging_dataset_copy_chipped/phase9t_staging_patch_review_template.csv`
- `artifacts/phase9t_manual_review_staging_dataset_copy_chipped/phase9t_staging_patch_review_checklist.md`
- `artifacts/phase9t_manual_review_staging_dataset_copy_chipped/phase9t_staging_dataset_integrity_check.csv`
- `artifacts/phase9t_manual_review_staging_dataset_copy_chipped/phase9t_non_mutation_manifest.json`
- `artifacts/phase9t_manual_review_staging_dataset_copy_chipped/visual_review_previews/error_0128_staging_label_overlay.jpg`
- `artifacts/phase9t_manual_review_staging_dataset_copy_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/prepare_phase9t_manual_review_staging_dataset_copy.py`

Run summary:

- created a manual review package for the Phase 9S staging dataset copy
- `status = phase9t_manual_review_package_created`
- `patched_case_ids = error_0128`
- `blocked_case_ids = error_0064`
- `staging_dataset_checked = true`
- `training_allowed = false`
- `approved_for_training = false`
- `human_review_required = true`
- `human_review_completed = false`
- `next_allowed_step = phase9u_human_training_gate_decision_for_staging_dataset_copy`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original dataset files were modified
- no new relabel patch was applied
- no staging label was edited during Phase 9T review packaging
- XAI remained support signal only and was not used as ground truth

## 2026-06-25 - Phase 9S apply relabel patch to staging dataset copy only for Chipped after human approval

Files created:

- `scripts/apply_phase9s_relabel_patch_to_staging_dataset_copy.py`
- `docs/phase9s_staging_dataset_relabel_patch_chipped.md`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_summary.json`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_summary.csv`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_patch_application_report.csv`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_original_vs_staging_label_diff.csv`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_staging_dataset_manifest.json`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_staging_dataset_integrity_check.csv`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_dataset_config_copy_check.csv`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_non_mutation_manifest.json`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_training_gate_checklist.md`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_dataset_application_approval_template.csv`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_dataset_application_approval_filled.csv`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_copy/`
- `artifacts/phase9s_staging_dataset_relabel_patch_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/apply_phase9s_relabel_patch_to_staging_dataset_copy.py --phase9r-output artifacts/phase9r_manual_review_materialized_patch_chipped --phase9q-output artifacts/phase9q_materialized_relabel_patch_chipped --dataset-config configs/dataset/drill_bit_yolo.yaml --output artifacts/phase9s_staging_dataset_relabel_patch_chipped --focus-class Chipped`
- `PYTHONPATH=src .venv/bin/python scripts/apply_phase9s_relabel_patch_to_staging_dataset_copy.py --phase9r-output artifacts/phase9r_manual_review_materialized_patch_chipped --phase9q-output artifacts/phase9q_materialized_relabel_patch_chipped --dataset-config configs/dataset/drill_bit_yolo.yaml --human-dataset-application-approval-file artifacts/phase9s_staging_dataset_relabel_patch_chipped/phase9s_dataset_application_approval_filled.csv --output artifacts/phase9s_staging_dataset_relabel_patch_chipped --focus-class Chipped`

Run summary:

- first run without approval stayed blocked with `status = phase9s_blocked_waiting_human_dataset_application_approval`
- blocked run kept `staging_dataset_created = false`
- blocked run kept `relabel_patch_applied_to_staging_copy = false`
- blocked run kept `original_dataset_mutated = false`
- blocked run kept `training_allowed = false`
- second run with explicit approval produced `status = relabel_patch_applied_to_staging_dataset_copy_only`
- `patched_case_ids = error_0128`
- `blocked_case_ids = error_0064`
- `staging_dataset_created = true`
- `relabel_patch_applied_to_staging_copy = true`
- `relabel_patch_applied_to_original_dataset = false`
- `original_dataset_mutated = false`
- `dataset_mutation_allowed = false`
- `training_allowed = false`
- `next_allowed_step = phase9t_manual_review_staging_dataset_copy_before_training_gate`
- staging images were mirrored as symlinks and labels were copied as regular files before applying the single staging-only relabel patch

Guardrails preserved:

- no original label file under `data/` was modified
- no annotation overwrite was performed on the original dataset
- no dataset config was changed
- no training was run
- no evaluation was run
- no inference was run
- no architecture, loss, or train-config changes were made
- `error_0064` remained blocked and was not patched
- XAI remained support signal only and was not used as ground truth
- no training-ready claim was made at Phase 9S

## 2026-06-25 - Phase 9R manual review of materialized copy-only relabel patch for Chipped before any dataset application

Files created:

- `scripts/review_phase9r_materialized_patch_before_dataset_application.py`
- `docs/phase9r_manual_review_materialized_patch_chipped.md`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_review_summary.json`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_review_summary.csv`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_materialized_patch_review.csv`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_label_copy_integrity_check.csv`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_dataset_application_approval_checklist.md`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_dataset_application_approval_template.csv`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_non_mutation_manifest.json`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_visual_review_previews/error_0128_original_image_copy.jpg`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_visual_review_previews/error_0128_old_bbox_overlay.jpg`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_visual_review_previews/error_0128_new_bbox_overlay.jpg`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/phase9r_visual_review_previews/error_0128_old_vs_new_bbox_overlay.jpg`
- `artifacts/phase9r_manual_review_materialized_patch_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src MPLCONFIGDIR=/tmp/mplconfig_phase9r .venv/bin/python scripts/review_phase9r_materialized_patch_before_dataset_application.py --phase9q-output artifacts/phase9q_materialized_relabel_patch_chipped --dataset-config configs/dataset/drill_bit_yolo.yaml --output artifacts/phase9r_manual_review_materialized_patch_chipped --focus-class Chipped`

Run summary:

- created manual review artifacts for the materialized copy-only patch from Phase 9Q
- `status = materialized_patch_manual_review_created`
- `reviewed_case_ids = error_0128`
- `blocked_case_ids = error_0064`
- `label_copy_integrity_passed = true`
- `materialized_patch_is_copy_only = true`
- `dataset_application_ready = false`
- `dataset_application_approval_required = true`
- `relabel_patch_applied_to_dataset = false`
- `original_dataset_mutated = false`
- `dataset_mutation_allowed = false`
- `training_allowed = false`
- `next_allowed_step = phase9s_apply_relabel_patch_to_staging_dataset_copy_only_after_human_approval`
- visual review previews were generated from old/new bbox overlays only

Guardrails preserved:

- no original label file under `data/` was modified
- no annotation overwrite was performed
- no staging dataset was created
- no dataset config was changed
- no training was run
- no evaluation was run
- no inference was run
- no architecture, loss, or train-config changes were made
- `error_0064` remained blocked and was not materialized
- XAI remained support signal only and was not used as ground truth

## 2026-06-25 - Phase 9Q copy-only relabel patch materialization for Chipped after explicit human final approval

Files created:

- `scripts/materialize_phase9q_relabel_patch_from_approved_dry_run.py`
- `docs/phase9q_materialized_relabel_patch_chipped.md`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_summary.json`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_summary.csv`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_human_final_approval_template.csv`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_human_final_approval_filled.csv`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_materialized_patch_spec.csv`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_label_patch_manifest.json`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_non_mutation_manifest.json`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_patch_application_instructions.md`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_materialized_labels/S271_Image__2025-11-17__11-16-07_bright_4_crop_6.txt`
- `artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_original_label_backup/S271_Image__2025-11-17__11-16-07_bright_4_crop_6.txt`
- `artifacts/phase9q_materialized_relabel_patch_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/materialize_phase9q_relabel_patch_from_approved_dry_run.py --phase9p-output artifacts/phase9p_relabel_patch_dry_run_review_chipped --dataset-config configs/dataset/drill_bit_yolo.yaml --output artifacts/phase9q_materialized_relabel_patch_chipped --focus-class Chipped`
- `PYTHONPATH=src .venv/bin/python scripts/materialize_phase9q_relabel_patch_from_approved_dry_run.py --phase9p-output artifacts/phase9p_relabel_patch_dry_run_review_chipped --dataset-config configs/dataset/drill_bit_yolo.yaml --human-final-approval-file artifacts/phase9q_materialized_relabel_patch_chipped/phase9q_human_final_approval_filled.csv --output artifacts/phase9q_materialized_relabel_patch_chipped --focus-class Chipped`

Run summary:

- first safe run without approval stayed blocked with `status = phase9q_blocked_waiting_human_final_approval`
- the blocked run created `phase9q_human_final_approval_template.csv` and did not materialize any label
- second run with explicit approval produced `status = relabel_patch_materialized_as_copy_only`
- `candidate_case_ids = error_0128`
- `blocked_case_ids = error_0064`
- `relabel_patch_materialized = true`
- `relabel_patch_applied_to_dataset = false`
- `original_dataset_mutated = false`
- `dataset_mutation_allowed = false`
- `training_allowed = false`
- `next_allowed_step = phase9r_manual_review_materialized_patch_before_any_dataset_application`
- only one copy-only materialized label was created for `error_0128`

Guardrails preserved:

- no original label file under `data/` was modified
- no annotation overwrite was performed
- no dataset config was changed
- no training was run
- no evaluation was run
- no inference was run
- no architecture or loss changes were made
- `error_0064` remained blocked for `needs_more_visual_evidence`

## 2026-06-25 - Phase 9L relabel patch proposal planning only for Chipped

Files created:

- `scripts/prepare_phase9l_relabel_patch_proposal_planning.py`
- `docs/phase9l_relabel_patch_proposal_planning_chipped.md`
- `artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_relabel_patch_proposal_table.csv`
- `artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_case_policy_action_plan.csv`
- `artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_patch_candidate_review_checklist.md`
- `artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_patch_proposal_schema.csv`
- `artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_non_mutation_manifest.json`
- `artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_relabel_patch_proposal_summary.json`
- `artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_relabel_patch_proposal_summary.csv`
- `artifacts/phase9l_relabel_patch_proposal_planning_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9l_relabel_patch_proposal_planning.py --phase9k-summary artifacts/phase9k_human_approval_guideline_v2_chipped/phase9k_guideline_v2_approval_summary.json --phase9k-policy-gate artifacts/phase9k_human_approval_guideline_v2_chipped/phase9k_policy_use_gate.csv --phase9i-guideline-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_chipped_guideline_v2_draft.md --phase9i-rules-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_chipped_decision_rules_v2.csv --phase9i-mapping-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_case_policy_mapping_v2.csv --phase9j-dryrun artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_revised_audit_case_policy_dryrun.csv --phase9j-gap-closure artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_gap_closure_review.csv --phase9j-summary artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_revised_guideline_dryrun_summary.json --output artifacts/phase9l_relabel_patch_proposal_planning_chipped --focus-class Chipped`

Run summary:

- created a relabel patch proposal planning package for `error_0064` and `error_0128`
- `status = relabel_patch_proposal_planning_created`
- `case_count = 2`
- `guideline_v2_policy_approved = true`
- `relabel_patch_proposal_created = true`
- `relabel_patch_applied = false`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `relabel_patch_allowed = false`
- `dataset_mutation_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = human_review_of_relabel_patch_proposal_required`

Guardrails preserved:

- Phase 9L remained proposal-only
- no relabel patch was applied
- no dataset files were modified
- no YOLO label files were modified
- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no architecture changes were made
- no loss changes were made
- training remained locked

## 2026-06-25 - Phase 9M final human review package for relabel patch proposals in Chipped

Files created:

- `scripts/prepare_phase9m_final_human_patch_review.py`
- `docs/phase9m_final_human_patch_review_chipped.md`
- `artifacts/phase9m_final_human_patch_review_chipped/phase9m_human_patch_review_template.csv`
- `artifacts/phase9m_final_human_patch_review_chipped/phase9m_patch_review_context_table.csv`
- `artifacts/phase9m_final_human_patch_review_chipped/phase9m_human_patch_review_checklist.md`
- `artifacts/phase9m_final_human_patch_review_chipped/phase9m_review_decision_schema.csv`
- `artifacts/phase9m_final_human_patch_review_chipped/phase9m_non_mutation_manifest.json`
- `artifacts/phase9m_final_human_patch_review_chipped/phase9m_final_human_patch_review_summary.json`
- `artifacts/phase9m_final_human_patch_review_chipped/phase9m_final_human_patch_review_summary.csv`
- `artifacts/phase9m_final_human_patch_review_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9m_final_human_patch_review.py --phase9l-summary artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_relabel_patch_proposal_summary.json --phase9l-proposal-table artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_relabel_patch_proposal_table.csv --phase9l-action-plan artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_case_policy_action_plan.csv --phase9l-checklist artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_patch_candidate_review_checklist.md --phase9l-manifest artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_non_mutation_manifest.json --output artifacts/phase9m_final_human_patch_review_chipped --focus-class Chipped`

Run summary:

- created a final human review package for Phase 9L patch proposals `error_0064` and `error_0128`
- `status = final_human_patch_review_package_created`
- `human_review_completed = false`
- `pending_human_review_count = 2`
- `relabel_patch_applied = false`
- `training_allowed = false`
- `relabel_patch_allowed = false`
- `dataset_mutation_allowed = false`
- `next_allowed_step = complete_human_review_template_before_patch_materialization_planning`

Guardrails preserved:

- Phase 9M remained final-human-review-package-only
- no relabel patch was applied
- no dataset files were modified
- no YOLO label files were modified
- no new label files were created
- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- XAI remained support evidence only
- training remained locked

## 2026-06-25 - Phase 9M full final review context package for Chipped

Files created:

- `scripts/package_phase9m_full_review_context.py`
- `phase9m_full_final_review_context_chipped.zip`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/package_phase9m_full_review_context.py --output phase9m_full_final_review_context_chipped.zip`
- `unzip -l phase9m_full_final_review_context_chipped.zip | head -n 80`
- `unzip -l phase9m_full_final_review_context_chipped.zip | grep -E "data/|dataset/|datasets/|runs/|experiments/|.venv|\\.pt|\\.pth|\\.onnx" || true`

Run summary:

- created a full review-context zip for Phase 9M final human patch review of `error_0064` and `error_0128`
- package included only allowlisted review context paths plus a package manifest
- package excluded dataset roots, training outputs, virtualenv files, and model checkpoints

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no YOLO label files were modified
- no relabel patch was applied
- no dataset files were modified
- no training gate was opened

## 2026-06-25 - Phase 9M.5 validate final human patch review for Chipped

Files created:

- `scripts/validate_phase9m_final_human_patch_review.py`
- `docs/phase9m_5_validate_final_human_patch_review_chipped.md`
- `artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_human_patch_review_filled.csv`
- `artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_human_patch_review_validation_report.csv`
- `artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_human_patch_review_validation_report.json`
- `artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_5_decision_summary.csv`
- `artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_5_decision_summary.json`
- `artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_5_gate_status.csv`
- `artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_5_non_mutation_manifest.json`
- `artifacts/phase9m_5_validated_human_patch_review_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/validate_phase9m_final_human_patch_review.py --phase9m-template artifacts/phase9m_final_human_patch_review_chipped/phase9m_human_patch_review_template.csv --phase9m-context artifacts/phase9m_final_human_patch_review_chipped/phase9m_patch_review_context_table.csv --phase9m-schema artifacts/phase9m_final_human_patch_review_chipped/phase9m_review_decision_schema.csv --phase9m-summary artifacts/phase9m_final_human_patch_review_chipped/phase9m_final_human_patch_review_summary.json --phase9m-manifest artifacts/phase9m_final_human_patch_review_chipped/phase9m_non_mutation_manifest.json --phase9l-summary artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_relabel_patch_proposal_summary.json --output artifacts/phase9m_5_validated_human_patch_review_chipped --focus-class Chipped --reviewer-id thanhmay_manual_review --review-date 2026-06-25`

Run summary:

- validated the human-provided final review decisions for `error_0064` and `error_0128`
- `status = final_human_patch_review_validated`
- `human_review_completed = true`
- `approved_patch_count = 1`
- `needs_more_visual_evidence_count = 1`
- `pending_human_review_count = 0`
- `approved_patch_case_ids = error_0128`
- `blocked_case_ids = error_0064`
- `relabel_patch_applied = false`
- `training_allowed = false`
- `dataset_mutation_allowed = false`
- `next_allowed_step = phase9n_materialization_planning_for_human_approved_patch_candidates_only`

Guardrails preserved:

- original Phase 9M template was not modified
- no relabel patch was applied
- no dataset files were modified
- no YOLO label files were modified
- no new label files were created
- no training was run
- no evaluation was run
- no inference was run
- no automatic bbox shrink or expansion was performed
- XAI remained support evidence only

## 2026-06-25 - Phase 9N materialization planning only for human-approved patch candidates in Chipped

Files created:

- `scripts/prepare_phase9n_materialization_planning.py`
- `docs/phase9n_materialization_planning_chipped.md`
- `artifacts/phase9n_materialization_planning_chipped/phase9n_materialization_candidate_plan.csv`
- `artifacts/phase9n_materialization_planning_chipped/phase9n_human_bbox_coordinate_template.csv`
- `artifacts/phase9n_materialization_planning_chipped/phase9n_blocked_cases.csv`
- `artifacts/phase9n_materialization_planning_chipped/phase9n_materialization_gate_status.csv`
- `artifacts/phase9n_materialization_planning_chipped/phase9n_non_mutation_manifest.json`
- `artifacts/phase9n_materialization_planning_chipped/phase9n_materialization_planning_summary.json`
- `artifacts/phase9n_materialization_planning_chipped/phase9n_materialization_planning_summary.csv`
- `artifacts/phase9n_materialization_planning_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9n_materialization_planning.py --phase9m5-filled artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_human_patch_review_filled.csv --phase9m5-summary artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_5_decision_summary.json --phase9m5-gate artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_5_gate_status.csv --phase9m5-manifest artifacts/phase9m_5_validated_human_patch_review_chipped/phase9m_5_non_mutation_manifest.json --phase9l-proposal-table artifacts/phase9l_relabel_patch_proposal_planning_chipped/phase9l_relabel_patch_proposal_table.csv --phase9m-context artifacts/phase9m_final_human_patch_review_chipped/phase9m_patch_review_context_table.csv --output artifacts/phase9n_materialization_planning_chipped --focus-class Chipped`

Run summary:

- created a materialization planning package only for human-approved patch candidates
- `status = materialization_planning_created`
- `approved_planning_case_ids = error_0128`
- `blocked_case_ids = error_0064`
- `patch_materialization_ready = false`
- `relabel_patch_applied = false`
- `dataset_mutation_allowed = false`
- `training_allowed = false`
- `next_allowed_step = collect_human_defined_bbox_coordinates_for_error_0128`

Guardrails preserved:

- no patch was materialized
- no dataset files were modified
- no YOLO label files were modified
- no new label files were created
- no annotation overwrite was performed
- no training was run
- no evaluation was run
- no inference was run
- no automatic bbox shrink or expansion was performed
- XAI remained `support_signal_only_not_ground_truth`

## 2026-06-25 - Phase 9O error_0128 coordinate review input package

Status: coordinate_review_input_package_created

Summary:
- Collected available visual/context evidence for `error_0128`.
- Kept `error_0064` blocked.
- Did not fill human-defined bbox coordinates.
- Did not apply relabel patch.
- Did not modify dataset files or YOLO label files.
- Did not run training, evaluation, or inference.
- XAI remained `support_signal_only_not_ground_truth`.

Output:
- `artifacts/phase9o_error0128_coordinate_review_input_chipped/`
- `phase9o_error0128_coordinate_review_input_chipped.zip`

## 2026-06-25 - Phase 9O human bbox coordinate filled validation rerun for Chipped

Status: human_bbox_coordinates_validated_dry_run_ready

Summary:
- Filled human-confirmed bbox coordinates for `error_0128`.
- Kept `error_0064` blocked.
- Reran Phase 9O coordinate validator.
- `human_bbox_coordinates_present = true`.
- `bbox_coordinates_valid = true`.
- `dry_run_patch_ready = true`.
- `patch_materialization_ready = false`.
- `next_allowed_step = phase9p_relabel_patch_dry_run_review_before_materialization`.

Confirmed coordinates:
- `new_class_id = 1`
- `new_x_center = 0.370536`
- `new_y_center = 0.803571`
- `new_width = 0.098214`
- `new_height = 0.267857`
- Pixel reference on 448x448 image: `x_min=144, y_min=300, x_max=188, y_max=420`

Guardrails:
- No relabel patch was applied.
- No dataset files were modified.
- No YOLO label files were modified.
- No new label files were created in the dataset.
- No annotation overwrite was performed.
- No training, evaluation, or inference was run.
- No automatic bbox adjustment was performed.
- XAI remained `support_signal_only_not_ground_truth`.

## 2026-06-25 - Phase 9P relabel patch dry-run review before materialization for Chipped

Status: relabel_patch_dry_run_review_created

Summary:
- Created a dry-run relabel patch review package for `error_0128`.
- Kept `error_0064` blocked.
- Compared the current YOLO label against the human-confirmed Phase 9O proposal.
- Required explicit final human approval before any Phase 9Q materialization step.
- `patch_materialization_ready = false`.
- `materialization_approval_required = true`.
- `relabel_patch_applied = false`.
- `dataset_mutation_allowed = false`.
- `training_allowed = false`.
- `next_allowed_step = phase9q_materialize_relabel_patch_only_after_human_final_approval`.

Guardrails:
- No relabel patch was materialized.
- No dataset files were modified.
- No YOLO label files were modified.
- No new label files were created in the dataset.
- No annotation overwrite was performed.
- No training, evaluation, or inference was run.
- No architecture or loss changes were made.
- XAI remained `support_signal_only_not_ground_truth`.

## 2026-06-25 - Phase 9K human approval of revised Chipped guideline v2

Files created:

- `scripts/prepare_phase9k_human_approval_guideline_v2.py`
- `docs/phase9k_human_approval_guideline_v2_chipped.md`
- `artifacts/phase9k_human_approval_guideline_v2_chipped/phase9k_human_approval_checklist.md`
- `artifacts/phase9k_human_approval_guideline_v2_chipped/phase9k_human_approval_template.csv`
- `artifacts/phase9k_human_approval_guideline_v2_chipped/phase9k_guideline_v2_approval_summary.json`
- `artifacts/phase9k_human_approval_guideline_v2_chipped/phase9k_guideline_v2_approval_summary.csv`
- `artifacts/phase9k_human_approval_guideline_v2_chipped/phase9k_policy_use_gate.csv`
- `artifacts/phase9k_human_approval_guideline_v2_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9k_human_approval_guideline_v2.py --phase9i-guideline-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_chipped_guideline_v2_draft.md --phase9i-rules-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_chipped_decision_rules_v2.csv --phase9i-mapping-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_case_policy_mapping_v2.csv --phase9j-checklist artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_guideline_v2_review_checklist.md --phase9j-dryrun artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_revised_audit_case_policy_dryrun.csv --phase9j-gap-closure artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_gap_closure_review.csv --phase9j-summary artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_revised_guideline_dryrun_summary.json --output artifacts/phase9k_human_approval_guideline_v2_chipped --focus-class Chipped`

Run summary:

- created a human approval package for revised guideline v2 covering `error_0064` and `error_0128`
- `status = revised_guideline_v2_approval_pending`
- `approval_status = pending_human_confirmation`
- `approved_for_policy_use = false`
- `approved_for_relabel_patch_planning = false`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `relabel_patch_allowed = false`
- `dataset_mutation_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = collect_human_approval_for_guideline_v2`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no relabel patch was created or applied
- no dataset changes were made
- no architecture changes were made
- no loss changes were made
- training remained locked

## 2026-06-25 - Phase 9J revised guideline v2 dry-run review for Chipped

Files created:

- `scripts/prepare_phase9j_revised_guideline_dryrun_review.py`
- `docs/phase9j_revised_guideline_dryrun_review_chipped.md`
- `artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_guideline_v2_review_checklist.md`
- `artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_revised_audit_case_policy_dryrun.csv`
- `artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_gap_closure_review.csv`
- `artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_case_decision_schema_v2.csv`
- `artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_revised_guideline_dryrun_summary.json`
- `artifacts/phase9j_revised_guideline_dryrun_review_chipped/phase9j_revised_guideline_dryrun_summary.csv`
- `artifacts/phase9j_revised_guideline_dryrun_review_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9j_revised_guideline_dryrun_review.py --phase9i-guideline-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_chipped_guideline_v2_draft.md --phase9i-rules-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_chipped_decision_rules_v2.csv --phase9i-mapping-v2 artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_case_policy_mapping_v2.csv --phase9i-resolution-table artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_revision_resolution_table.csv --phase9i-summary artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_revision_summary.json --phase9g-gap-report artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_gap_report.csv --phase9g-dryrun artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_audit_case_policy_dryrun.csv --phase9h-revision-items artifacts/phase9h_guideline_approval_decision_chipped/phase9h_guideline_revision_items.csv --phase9h-summary artifacts/phase9h_guideline_approval_decision_chipped/phase9h_guideline_approval_summary.json --output artifacts/phase9j_revised_guideline_dryrun_review_chipped --focus-class Chipped`

Run summary:

- completed revised dry-run review for `error_0064` and `error_0128`
- `case_count = 2`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `relabel_patch_allowed = false`
- `dataset_mutation_allowed = false`
- `intervention_training_ready = false`
- status and next step depended on Phase 9J gap-closure review results

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no relabel patch was created or applied
- no dataset changes were made
- no architecture changes were made
- no loss changes were made
- training remained locked

## 2026-06-26 - Phase 9X explicit staging training execution approval gate for Chipped

Files created:

- `scripts/collect_phase9x_training_execution_approval.py`
- `docs/phase9x_training_execution_approval_gate_chipped.md`
- `artifacts/phase9x_training_execution_approval_chipped/phase9x_training_execution_approval_summary.json`
- `artifacts/phase9x_training_execution_approval_chipped/phase9x_training_execution_approval_template.csv`
- `artifacts/phase9x_training_execution_approval_chipped/phase9x_training_execution_approval_used.csv`
- `artifacts/phase9x_training_execution_approval_chipped/phase9x_execution_gate_checklist.md`
- `artifacts/phase9x_training_execution_approval_chipped/phase9x_non_execution_manifest.json`
- `artifacts/phase9x_training_execution_approval_chipped/phase9x_execution_gate_review.csv`
- `artifacts/phase9x_training_execution_approval_chipped/phase9x_training_execution_approval_approved.csv`
- `artifacts/phase9x_training_execution_approval_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `.venv/bin/python scripts/collect_phase9x_training_execution_approval.py`
- `.venv/bin/python scripts/collect_phase9x_training_execution_approval.py --approval-csv artifacts/phase9x_training_execution_approval_chipped/phase9x_training_execution_approval_approved.csv`

Run summary:

- verified the live Phase 9W summary remained `phase9w_final_command_review_prepared`
- verified the live Phase 9V summary remained `phase9v_staging_training_config_prepared`
- verified the Phase 9V config still pointed to the Phase 9S staging dataset yaml
- verified the command preview still matched the reviewed Phase 9V config lineage
- default run produced `status = phase9x_training_execution_approval_pending`
- approved-input run produced `status = phase9x_training_execution_approved`
- neither run executed training, evaluation, or inference
- neither run mutated the original dataset or the staging dataset copy

Guardrails preserved:

- no training command was executed
- no training was run
- no evaluation was run
- no inference was run
- no original dataset was modified
- no staging dataset mutation was performed by Phase 9X
- no architecture changes were made
- no loss changes were made

## 2026-06-25 - Phase 9I Chipped guideline revision pass 1

Files created:

- `scripts/prepare_phase9i_chipped_guideline_revision_pass1.py`
- `docs/phase9i_chipped_guideline_revision_pass1.md`
- `artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_chipped_guideline_v2_draft.md`
- `artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_revision_resolution_table.csv`
- `artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_chipped_decision_rules_v2.csv`
- `artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_case_policy_mapping_v2.csv`
- `artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_revision_summary.json`
- `artifacts/phase9i_chipped_guideline_revision_pass1/phase9i_revision_summary.csv`
- `artifacts/phase9i_chipped_guideline_revision_pass1/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9i_chipped_guideline_revision_pass1.py --phase9f-guideline artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_chipped_annotation_guideline_draft.md --phase9f-rules artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_chipped_decision_rules.csv --phase9f-mapping artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_case_policy_mapping.csv --phase9g-gap-report artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_gap_report.csv --phase9g-dryrun artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_audit_case_policy_dryrun.csv --phase9h-revision-items artifacts/phase9h_guideline_approval_decision_chipped/phase9h_guideline_revision_items.csv --phase9h-case-review artifacts/phase9h_guideline_approval_decision_chipped/phase9h_case_level_approval_review.csv --phase9h-summary artifacts/phase9h_guideline_approval_decision_chipped/phase9h_guideline_approval_summary.json --output artifacts/phase9i_chipped_guideline_revision_pass1 --focus-class Chipped`

Run summary:

- created a revised Chipped guideline v2 draft focused on `error_0064` and `error_0128`
- `status = guideline_revision_pass_created`
- `guideline_v2_draft_created = true`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `relabel_patch_allowed = false`
- `dataset_mutation_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = revised_guideline_dryrun_review_required`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no relabel patch was created or applied
- no dataset changes were made
- no architecture changes were made
- no loss changes were made
- training remained locked

## 2026-06-25 - Phase 9H human guideline approval or revision decision for Chipped

Files created:

- `scripts/prepare_phase9h_guideline_approval_decision.py`
- `docs/phase9h_guideline_approval_decision_chipped.md`
- `artifacts/phase9h_guideline_approval_decision_chipped/phase9h_human_guideline_approval_checklist.md`
- `artifacts/phase9h_guideline_approval_decision_chipped/phase9h_guideline_approval_decision_template.csv`
- `artifacts/phase9h_guideline_approval_decision_chipped/phase9h_case_level_approval_review.csv`
- `artifacts/phase9h_guideline_approval_decision_chipped/phase9h_guideline_revision_items.csv`
- `artifacts/phase9h_guideline_approval_decision_chipped/phase9h_guideline_approval_summary.json`
- `artifacts/phase9h_guideline_approval_decision_chipped/phase9h_guideline_approval_summary.csv`
- `artifacts/phase9h_guideline_approval_decision_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9h_guideline_approval_decision.py --phase9f-guideline artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_chipped_annotation_guideline_draft.md --phase9f-rules artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_chipped_decision_rules.csv --phase9f-mapping artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_case_policy_mapping.csv --phase9f-summary artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_guideline_revision_summary.json --phase9g-checklist artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_review_checklist.md --phase9g-dryrun artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_audit_case_policy_dryrun.csv --phase9g-decision-schema artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_case_decision_schema.csv --phase9g-gap-report artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_gap_report.csv --phase9g-summary artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_review_summary.json --output artifacts/phase9h_guideline_approval_decision_chipped --focus-class Chipped`

Run summary:

- created a Phase 9H approval-or-revision decision package for `error_0064` and `error_0128`
- `status = guideline_revision_required`
- `case_count = 2`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `relabel_patch_allowed = false`
- `dataset_mutation_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = guideline_revision_required`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no relabel patch was created or applied
- no dataset changes were made
- no architecture changes were made
- no loss changes were made
- training remained locked

## 2026-06-25 - Phase 9G guideline review and dry-run policy application for Chipped

Files created:

- `scripts/prepare_phase9g_guideline_review_dryrun.py`
- `docs/phase9g_guideline_review_dryrun_chipped.md`
- `artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_review_checklist.md`
- `artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_audit_case_policy_dryrun.csv`
- `artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_case_decision_schema.csv`
- `artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_gap_report.csv`
- `artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_review_summary.json`
- `artifacts/phase9g_guideline_review_dryrun_chipped/phase9g_guideline_review_summary.csv`
- `artifacts/phase9g_guideline_review_dryrun_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9g_guideline_review_dryrun.py --phase9f-guideline artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_chipped_annotation_guideline_draft.md --phase9f-rules artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_chipped_decision_rules.csv --phase9f-mapping artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_case_policy_mapping.csv --phase9f-summary artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_guideline_revision_summary.json --phase9e-case-outcomes artifacts/phase9e_second_review_outcome_chipped/phase9e_case_outcomes.csv --phase9e-summary artifacts/phase9e_second_review_outcome_chipped/phase9e_intervention_scope_summary.json --phase9d-confirmed artifacts/phase9d_second_review_chipped/phase9d_second_review_confirmed.csv --output artifacts/phase9g_guideline_review_dryrun_chipped --focus-class Chipped`

Run summary:

- completed a dry-run guideline review for `error_0064` and `error_0128`
- `status = guideline_review_dryrun_completed`
- `case_count = 2`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `relabel_patch_allowed = false`
- `dataset_mutation_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = human_guideline_approval_or_policy_revision`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no relabel patch was created or applied
- no dataset changes were made
- no architecture changes were made
- no loss changes were made
- training remained locked

## 2026-06-25 - Phase 9F annotation guideline revision draft for Chipped failure cases

Files created:

- `scripts/prepare_phase9f_annotation_guideline_revision.py`
- `docs/phase9f_annotation_guideline_revision_chipped.md`
- `artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_chipped_annotation_guideline_draft.md`
- `artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_chipped_decision_rules.csv`
- `artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_case_policy_mapping.csv`
- `artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_guideline_revision_summary.json`
- `artifacts/phase9f_annotation_guideline_revision_chipped/phase9f_guideline_revision_summary.csv`
- `artifacts/phase9f_annotation_guideline_revision_chipped/README.md`

Files modified:

- `docs/experiment_log.md`

Commands run:

- `.venv/bin/python -m compileall src scripts`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_phase9f_annotation_guideline_revision.py --phase9e-case-outcomes artifacts/phase9e_second_review_outcome_chipped/phase9e_case_outcomes.csv --phase9e-policy-update-cases artifacts/phase9e_second_review_outcome_chipped/phase9e_policy_update_cases.csv --phase9e-summary artifacts/phase9e_second_review_outcome_chipped/phase9e_intervention_scope_summary.json --phase9d-confirmed artifacts/phase9d_second_review_chipped/phase9d_second_review_confirmed.csv --dataset-config configs/dataset/drill_bit_yolo.yaml --output artifacts/phase9f_annotation_guideline_revision_chipped --focus-class Chipped`

Run summary:

- created a conservative guideline draft for `Chipped` without changing annotations
- `status = annotation_guideline_revision_draft_created`
- `policy_update_case_count = 2`
- `policy_update_case_ids = error_0064,error_0128`
- `guideline_revision_draft_created = true`
- `decision_rules_created = true`
- `case_policy_mapping_created = true`
- `annotation_policy_update_required = true`
- `training_allowed = false`
- `auto_relabel_allowed = false`
- `relabel_patch_allowed = false`
- `intervention_training_ready = false`
- `next_allowed_step = guideline_review_or_policy_application_to_audit_cases`

Guardrails preserved:

- no training was run
- no evaluation was run
- no inference was run
- no original annotations were modified
- no new labels were created
- no relabel patch was created or applied
- no dataset changes were made
- no architecture changes were made
- no loss changes were made
- training remained locked
