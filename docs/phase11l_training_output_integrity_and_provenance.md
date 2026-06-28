# Phase 11L Training Output Integrity And Provenance

Phase 11L is a strict non-execution validation phase that follows Phase 11K. Phase 11K collected and summarized the local Kaggle training outputs. Phase 11L exists to check whether those collected outputs are internally consistent, traceable, and acceptable as a baseline trained checkpoint candidate before any later prepare-only evaluation planning.

Purpose:

1. Validate that the key Phase 11K artifacts exist and contain the required fields.
2. Re-inspect the referenced local `results.csv`, `best.pt`, and `last.pt` files directly.
3. Confirm that direct parsing of `results.csv` matches the metrics recorded by Phase 11K.
4. Confirm that checkpoint file metadata remains consistent with the Phase 11K weight manifest.
5. Record the provenance caveat conservatively before any later evaluation preparation phase.

What Phase 11L validates:

- Phase 11K summary, metrics CSV, and weight manifest availability.
- `results.csv` readability, row count, epoch monotonicity, and required metric column presence.
- Direct best and final metrics for:
  - `metrics/mAP50-95(B)`
  - `metrics/mAP50(B)`
- Alignment between direct parsing and the Phase 11K recorded values with a small numeric tolerance.
- Checkpoint path existence, non-empty file size, modified time, and streamed `sha256` hash matching against the Phase 11K manifest.
- Descriptive post-best diagnostics such as degradation or plateau, without making any broader research claim.

What Phase 11L does not do:

- no training
- no evaluation
- no inference
- no prediction
- no export
- no dataset mutation
- no relabeling
- no checkpoint loading into tensors
- no copying `.pt` files into artifacts

Provenance caveat:

- The repo-local Phase 11J.1 summary still reports `phase11j1_execution_not_started_missing_execute_flag`.
- Phase 11K therefore relied on direct local training output inspection rather than a repo-local execution-success summary.
- Phase 11L preserves that caveat explicitly.
- As a result, Phase 11L can validate the integrity and consistency of the local files, but it cannot prove how Kaggle originally produced them.
- A passing Phase 11L gate therefore accepts `best.pt` only as a `baseline trained checkpoint candidate`, not as a final research result.

Why `.pt` files are not committed or copied:

- The weight files are large binary outputs that should remain outside normal Git tracking.
- Phase 11L records path, size, modified time, and `sha256` only.
- This keeps provenance artifacts small and reproducible while avoiding accidental checkpoint duplication.

Pass/fail gate meaning:

- Pass:
  - local outputs are internally consistent with Phase 11K
  - checkpoint metadata matches Phase 11K
  - the provenance caveat is recorded
  - `best.pt` may be accepted as `best_pt_for_phase11m_prepare_only_evaluation`
- Fail:
  - a required metadata artifact is missing
  - direct parsing does not match Phase 11K
  - checkpoint metadata does not match
  - or the provenance record is incomplete

Next step after pass:

- `phase11m0_prepare_approved_test_evaluation_no_execution`
