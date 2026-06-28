# Phase 11P Final Test Evaluation Evidence Report

## Section 1: Scope and status

- Phase 11P status: `phase11p_final_report_prepared_with_metric_provenance_caveat`
- Scope: report-only consolidation for Phase 11J through Phase 11O.
- Inputs:
  - Phase 11N summary: `/home/thanhmay/workspace/XAI-small_object_detection/artifacts/phase11n_test_evaluation_output_collection_and_validation/phase11n_test_evaluation_output_summary.json`
  - Phase 11O summary: `/home/thanhmay/workspace/XAI-small_object_detection/artifacts/phase11o_manual_test_metric_review/phase11o_manual_metric_review_summary.json`
- No training, evaluation, inference, prediction, export, checkpoint loading, or metric recomputation was performed by Phase 11P.

## Section 2: Training output integrity summary

- The training output integrity record carried forward from Phase 11L remains available and reportable with provenance caveat.
- Best epoch from Phase 11L: `42.0`
- Best `mAP50-95` from Phase 11L: `0.36688`
- Final epoch from Phase 11L: `100.0`
- Final `mAP50-95` from Phase 11L: `0.35709`
- Final `mAP50` from Phase 11L: `0.71681`
- These are training-output metrics validated by local output integrity checks, not new test metrics from Phase 11M.1.

## Section 3: Test evaluation output inventory

- Evaluation directory from Phase 11N: `/home/thanhmay/workspace/XAI-small_object_detection/experiments/phase11m_test_eval/yolov8n_drill_bit_phase11m_test_eval`
- Training directory linked by Phase 11N: `/home/thanhmay/workspace/XAI-small_object_detection/experiments/phase11j_training`
- `predictions.json` available: `True`
- `predictions.json` size bytes: `7029583`
- `confusion_matrix.png` available: `True`
- `confusion_matrix_normalized.png` available: `True`
- `PR` curve available: `True`
- `F1` curve available: `True`
- `P` curve available: `True`
- `R` curve available: `True`
- Validation batch image count: `6`
- Evaluation outputs credible as files: `True`

## Section 4: Test metric provenance status

- Aggregate test metrics were not available from a parseable source and were not manually validated in Phase 11O.
- Evaluation outputs are credible as files, but aggregate metric reporting remains blocked.
- No numeric claim about test precision, recall, `mAP50`, or `mAP50-95` is made in this report.

## Section 5: What can be reported

- The Phase 11L training output integrity summary can be reported with provenance caveat.
- The Phase 11N evaluation output directory can be reported as credible by file inventory and artifact presence.
- The current final report can state that test evaluation outputs exist, but aggregate test metrics remain unavailable or not validated for reporting.

## Section 6: What cannot be claimed

- No wording such as `test mAP achieved ...` is allowed in the current state.
- No numeric test precision, recall, `mAP50`, or `mAP50-95` claim is allowed.
- This report does not prove Kaggle execution provenance beyond the carried-forward caveat.
- This report does not recompute any metric from `predictions.json` or from images, labels, or checkpoints.

## Section 7: Caveats and next recommended action

- Carried-forward provenance caveat: Phase 11K used direct local output inspection because the repo-local Phase 11J.1 summary still reports 'phase11j1_execution_not_started_missing_execute_flag'. Phase 11L therefore validates only the local training output integrity and cannot prove how Kaggle produced these files.
- The remaining metric caveat is that Phase 11M.1 did not preserve parseable aggregate test metric artifacts.
- Recommended next action: recover visible Kaggle notebook metrics into Phase 11O or keep using this report without numeric test metric claims.

## Section 8: Non-execution and non-mutation guarantees

- `training_executed = false`
- `evaluation_executed = false`
- `inference_executed = false`
- `prediction_executed = false`
- `export_executed = false`
- `checkpoint_loaded = false`
- `dataset_mutated = false`
- `labels_mutated = false`
- `weights_modified = false`
- `weights_copied_to_artifacts = false`
- `large_outputs_copied_to_artifacts = false`
- `predictions_json_used_for_metric_computation = false`
- `metrics_recomputed = false`

## Metric provenance table

| metric_name | value | source_phase | validation_status | reporting_status |
| --- | --- | --- | --- | --- |
| best_epoch | 42.0 | 11L via 11N | validated_by_phase11l_local_output_integrity | allowed_with_training_output_provenance_caveat |
| best_metric_map50_95 | 0.36688 | 11L via 11N | validated_by_phase11l_local_output_integrity | allowed_with_training_output_provenance_caveat |
| final_epoch | 100.0 | 11L via 11N | validated_by_phase11l_local_output_integrity | allowed_with_training_output_provenance_caveat |
| final_metric_map50_95 | 0.35709 | 11L via 11N | validated_by_phase11l_local_output_integrity | allowed_with_training_output_provenance_caveat |
| final_metric_map50 | 0.71681 | 11L via 11N | validated_by_phase11l_local_output_integrity | allowed_with_training_output_provenance_caveat |
| test_precision |   | 11O | not_validated | blocked |
| test_recall |   | 11O | not_validated | blocked |
| test_map50 |   | 11O | not_validated | blocked |
| test_map50_95 |   | 11O | not_validated | blocked |
