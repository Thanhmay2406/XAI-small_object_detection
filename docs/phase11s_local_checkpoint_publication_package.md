# Phase 11S Local Checkpoint Publication Package

Phase 11S prepares a local checkpoint publication package after Phase 11R.1 validates a real human decision for package preparation only.

Purpose:

1. Require the exact validated Phase 11R.1 package-preparation gate.
2. Prepare a local publication package bundle without enabling checkpoint publication.
3. Preserve metadata, provenance references, and metric caveats for later manual review.
4. Keep all execution-sensitive actions blocked.

Required inputs:

- `artifacts/phase11r1_checkpoint_publication_decision_validation/phase11r1_publication_decision_validation_summary.json`
- `artifacts/phase11r_checkpoint_publication_decision_gate/phase11r_checkpoint_inventory.csv`
- `artifacts/phase11r_checkpoint_publication_decision_gate/phase11r_publication_decision_template.csv`

Optional references:

- Phase 11R summary path carried in the Phase 11R.1 summary
- Phase 11P report path carried forward through Phase 11R if available
- Phase 11J.0, 11N, 11O, and 11P summaries if available

What Phase 11S does:

1. Verifies the Phase 11R.1 status is `phase11r1_checkpoint_publication_decision_validated_for_package_preparation`.
2. Requires `checkpoint_package_preparation_allowed = true`.
3. Keeps `checkpoint_publication_allowed = false`.
4. Optionally computes a SHA256 checksum by streaming raw bytes only when the decision CSV requests it.
5. Builds a local artifact bundle containing a release manifest, model-card draft, caveats, and non-execution record.
6. Defaults to a metadata-only package unless an explicit `include_checkpoint_binary_in_local_package=true` field is present in the decision CSV.

What Phase 11S never does:

- no checkpoint upload
- no checkpoint loading with `torch`, `ultralytics`, `YOLO`, or any model library
- no training
- no evaluation
- no inference
- no dataset mutation

Outputs:

- `artifacts/phase11s_local_checkpoint_publication_package/phase11s_publication_package_summary.json`
- `artifacts/phase11s_local_checkpoint_publication_package/phase11s_checkpoint_checksums.csv`
- `artifacts/phase11s_local_checkpoint_publication_package/phase11s_release_manifest.json`
- `artifacts/phase11s_local_checkpoint_publication_package/phase11s_model_card_draft.md`
- `artifacts/phase11s_local_checkpoint_publication_package/phase11s_publication_caveats.md`
- `artifacts/phase11s_local_checkpoint_publication_package/phase11s_non_execution_manifest.json`
- `artifacts/phase11s_local_checkpoint_publication_package/README.md`

Recommended success outcome:

- `status = phase11s_local_checkpoint_publication_package_prepared_metadata_only`
- `phase11r1_validated = true`
- `checkpoint_package_preparation_allowed = true`
- `checkpoint_publication_allowed = false`
- `checkpoint_upload_executed = false`
- `checkpoint_load_executed = false`
- `checkpoint_binary_copied = false`
- `next_allowed_step = phase11t_manual_checkpoint_publication_execution_gate_or_hold`
