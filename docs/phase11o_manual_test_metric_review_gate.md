# Phase 11O Manual Test Metric Review Gate

Phase 11O is a strict manual metric review gate for the existing Phase 11M.1 test evaluation outputs after Phase 11N concluded that the evaluation directory is credible but machine-readable aggregate metrics are still missing.

Purpose:

1. Read the Phase 11N summary and carry its provenance context forward.
2. Create a small CSV template for manual entry of test metrics.
3. Validate exactly one manually reviewed metric row when a filled CSV is provided.
4. Decide whether the metrics are review-only, rejected, or reporting-eligible with an explicit caveat flag.

Why Phase 11O exists:

- The credible evaluation directory exists under `experiments/phase11m_test_eval/yolov8n_drill_bit_phase11m_test_eval`.
- `predictions.json` exists, but this phase is not allowed to use it to recompute mAP, precision, or recall.
- `results.csv`, `results.json`, and `args.yaml` are absent from the evaluation directory, and the saved stdout log is not machine-parseable for aggregate metrics.
- A manual provenance step is therefore required before any later reporting phase can rely on test metrics.

Accepted manual metric sources:

- visible Kaggle notebook output from the Phase 11M.1 run
- copied Ultralytics console summary from the Phase 11M.1 run
- a saved manual note that was transcribed from that visible output
- another manual source only when the reviewer describes it explicitly

Why `predictions.json` is not used here:

- `predictions.json` is a per-prediction runtime artifact, not a trusted aggregate metric record for this gate.
- Recomputing mAP, precision, or recall from it would require logic outside the approved manual-review scope.
- That would blur the boundary between a review gate and a new evaluation or analysis phase.
- If recomputation is needed, it must happen in a separate explicitly approved phase.

Why manual metric provenance matters:

- Phase 11L and Phase 11N already carry a provenance caveat that repo-local artifacts prove output integrity but not the full Kaggle execution chain.
- Manual metric provenance makes the reporting caveat explicit instead of silently inferring results from incomplete machine-readable outputs.
- This keeps the workflow conservative: metrics can be reviewed and cited with caveat, or rejected until a cleaner rerun exists.

Strict validation rules:

- `phase` must be `11O` or `Phase 11O`
- exactly one data row is allowed
- `metric_source_type` must be one of:
  - `kaggle_notebook_visible_output`
  - `saved_manual_note`
  - `copied_ultralytics_console_summary`
  - `other_manual_source`
- `test_precision`, `test_recall`, `test_map50`, and `test_map50_95` must each be numeric in `[0, 1]`
- `source_confirms_test_split`, `source_confirms_model_best_pt`, and `source_confirms_dataset_yaml` must each be `true` or `false`
- `reviewer_decision` must be one of:
  - `approved_for_reporting_with_caveat`
  - `rejected_needs_rerun_or_better_logs`
  - `pending_manual_review`

Default behavior without a valid filled CSV:

- create the template CSV if it does not exist yet
- keep Phase 11O in `phase11o_manual_test_metric_review_pending`
- set `manual_metrics_available = false`
- set `test_metrics_validated = false`
- set `reporting_allowed = false`
- set `next_allowed_step = fill_phase11o_manual_metric_review_csv_from_kaggle_visible_output_then_rerun_phase11o`
- exit with code `0` because this is a normal gate-preparation outcome, not a runtime failure

Manual review outcomes:

- If the CSV is valid and the reviewer decision is `approved_for_reporting_with_caveat` but `--allow-reporting-with-caveat` is not provided:
  - metrics are validated
  - reporting remains blocked
  - the user must rerun Phase 11O with the explicit allow flag to unlock reporting
- If the CSV is valid and the reviewer decision is `approved_for_reporting_with_caveat` and the allow flag is present:
  - metrics are validated
  - reporting becomes allowed with caveat
  - the next step is Phase 11P final reporting preparation
- If the CSV is valid but the reviewer decision stays pending or rejected:
  - the metrics remain review-only
  - reporting remains blocked
  - later action depends on reviewer resolution or a cleaner rerun in a separate approved phase

What Phase 11O never does:

- no `yolo val`
- no inference or prediction execution
- no Ultralytics checkpoint loading in Python
- no opening or copying `best.pt`
- no metric recomputation from `predictions.json`
- no dataset YAML edits
- no image or label changes
- no copying large evaluation outputs into artifacts
