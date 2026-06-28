# Phase 11N Test Evaluation Output Collection And Validation

Phase 11N is a strict non-execution collector and validator for existing Phase 11M.1 test evaluation outputs.

Purpose:

1. Inspect the completed evaluation output directory under `experiments/phase11m_test_eval/yolov8n_drill_bit_phase11m_test_eval`.
2. Link those outputs back to the existing training output directory under `experiments/phase11j_training`.
3. Record file presence, file metadata, and any machine-readable metrics that can be parsed conservatively.
4. Produce a small metadata bundle without copying large evaluation outputs.

Why Phase 11N follows Phase 11M.1:

- Phase 11M.1 is the execution wrapper phase.
- Phase 11N is the follow-up collector/validator that inspects the resulting evaluation directory without re-running anything.

Inspected output directories:

- `experiments/phase11m_test_eval/yolov8n_drill_bit_phase11m_test_eval`
- `experiments/phase11j_training`

What Phase 11N validates:

- whether the evaluation directory exists and is non-empty
- whether common Ultralytics validation artifacts exist
- whether `predictions.json` exists and how large it is
- whether confusion matrices and curve plots exist
- whether validation batch preview images exist
- whether `args.yaml`, `results.csv`, `results.json`, or logs expose machine-readable metrics

What metrics were parsed:

- If `results.json`, `results.csv`, or Phase 11M.1 stdout contains parseable aggregate metrics, Phase 11N records them.
- If only plots, images, and `predictions.json` exist, Phase 11N does not infer aggregate metrics from those files.
- In that case the output is still considered credible, but `needs_manual_metric_review = true`.

What Phase 11N does not do:

- no evaluation
- no inference
- no prediction
- no training
- no export
- no dataset mutation
- no label mutation
- no checkpoint loading
- no checkpoint copying

Why large outputs and images are not copied:

- Evaluation outputs can be large and are runtime artifacts, not provenance metadata.
- Phase 11N records paths, sizes, mtimes, and hashes where reasonable.
- Images and large JSON outputs remain in the evaluation directory and are not duplicated into artifacts.

Carried-forward provenance caveat:

- the repo-local Phase 11J.1 summary originally remained a dry-run record
- Phase 11K relied on direct local output inspection
- Phase 11L validated training-output integrity only
- the checkpoint remains a baseline checkpoint candidate, not a final research result
- Phase 11N carries that caveat forward while validating only the collected evaluation outputs

Gate meanings:

- pass:
  - evaluation directory exists
  - credible evaluation artifacts are present
  - machine-readable metrics were parsed
- needs review:
  - evaluation directory exists
  - credible evaluation artifacts are present
  - machine-readable aggregate metrics were not parsed automatically
- fail:
  - evaluation directory is missing or empty
  - or the outputs do not look like a real evaluation bundle

Next step after pass:

- `phase11o_test_error_case_selection_and_xai_evidence_planning`
