# Phase 11P Final Test Evaluation Report With Metric Provenance Caveat

Phase 11P is a strict report-only consolidation phase for the Phase 11J through Phase 11O chain.

Purpose:

1. Read the Phase 11N evaluation-output summary.
2. Read the Phase 11O manual metric review summary.
3. Prepare a final evidence report that states exactly what is supported and what remains blocked.
4. Keep test metric wording conservative unless Phase 11O explicitly allows reporting.

Inputs:

- `artifacts/phase11n_test_evaluation_output_collection_and_validation/phase11n_test_evaluation_output_summary.json`
- `artifacts/phase11o_manual_test_metric_review/phase11o_manual_metric_review_summary.json`

Decision logic:

- If `Phase 11O reporting_allowed = true`:
  - Phase 11P may include the validated manual test metrics.
  - Status becomes `phase11p_final_report_prepared_with_validated_manual_test_metrics`.
  - The report must still say those metrics were manually extracted, not machine-parsed.
- If `Phase 11O reporting_allowed = false`:
  - Phase 11P must not include numeric test metric claims.
  - The report must state that aggregate test metrics are unavailable or not validated.
  - Status becomes `phase11p_final_report_prepared_with_metric_provenance_caveat`.

What Phase 11P is allowed to report:

- the Phase 11L training output integrity summary
- best epoch and final training metrics from Phase 11L
- the Phase 11N evaluation output inventory and credibility as files
- the carried-forward provenance caveat
- the current status of test metric availability and validation

What Phase 11P must not do:

- no training
- no evaluation
- no inference
- no prediction
- no export
- no checkpoint loading
- no metric recomputation
- no parsing `predictions.json` to derive aggregate test metrics
- no dataset mutation
- no label mutation
- no copying weights or large evaluation outputs into artifacts

Required safe wording when Phase 11O has not unlocked reporting:

- `Aggregate test metrics were not available from a parseable source and were not manually validated in Phase 11O.`
- `Evaluation outputs are credible as files, but aggregate metric reporting remains blocked.`

Required caveat even when Phase 11O does unlock reporting:

- aggregate test metrics are manually extracted and validated
- they are not machine-parsed from preserved `results.csv`, `results.json`, or stdout metrics
