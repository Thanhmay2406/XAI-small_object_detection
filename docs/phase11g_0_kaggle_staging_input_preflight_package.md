# Phase 11G.0 Kaggle Staging Input Preflight Package

Phase 11G.0 is the staging-input and Kaggle-preflight preparation phase for the approved Phase 11G training path. This phase is strictly prepare-only. It does not run training, evaluation, or inference, does not mutate the original dataset, and does not mutate the staging dataset copy.

The original `data/` tree is not the correct input for Phase 11G because the approved training path was built around the Phase 9S staging copy, not the untouched original dataset. The approved staging input is `artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_copy`, which is the controlled dataset scope referenced by the later approval and execution-package phases.

The local staging YAML at `artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml` is useful as a local reference, but it contains a local absolute path under `/home/thanhmay/...`. Kaggle cannot rely on that path, so Phase 11G.0 prepares a Kaggle YAML template with an explicit `/kaggle/input/REPLACE_WITH_STAGING_DATASET_SLUG/staging_dataset_copy` placeholder instead of pretending the local path will resolve remotely.

This phase checks that:

- Phase 11F remains in `phase11f_approved_staging_training_execution_package_prepared`
- Phase 11F.1 remains in `phase11f_1_kaggle_execution_adapter_package_prepared`
- Phase 11F.1 still routes to `phase11g_execute_approved_staging_training_on_kaggle`
- no upstream training execution has occurred
- the local staging dataset copy and staging YAML both exist
- the staging dataset copy contains the expected `images/*` and `labels/*` split directories

The package under `artifacts/phase11g_0_kaggle_staging_input_preflight_package/` is intended to help upload the correct staging dataset to Kaggle, rewrite the YAML safely for the Kaggle mount layout, and run a Kaggle-side preflight check before any real Phase 11G training command is allowed.

Real Phase 11G training remains out of scope for Phase 11G.0. The actual training command may only be considered after the staging dataset has been uploaded to Kaggle and the Kaggle preflight has passed.
