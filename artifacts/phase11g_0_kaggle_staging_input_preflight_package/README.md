# Phase 11G.0 Kaggle Staging Input Preflight Package

This package prepares Kaggle-side staging-input checks only. It does not run training, evaluation, or inference.

## Why this package exists
- Phase 11G must use the Phase 9S staging dataset copy, not the original `data/` dataset tree.
- The local staging YAML contains a local absolute path and cannot be used on Kaggle without rewriting.
- Kaggle preflight must pass before any real Phase 11G training command is allowed to run.

## Local staging inventory
- `train`: images=4219, labels=4219, directories_ready=true
- `valid`: images=916, labels=916, directories_ready=true
- `test`: images=906, labels=906, directories_ready=true

## Summary
- status: `phase11g_0_kaggle_staging_input_preflight_package_prepared`
- ready_for_kaggle_upload_of_staging_dataset: `true`
- ready_for_phase11g_training_after_kaggle_preflight: `false`
- next_allowed_step: `upload_phase9s_staging_dataset_copy_to_kaggle_and_run_phase11g_preflight`

## Files
- `phase11g_0_preflight_summary.json`
- `phase11g_0_staging_dataset_inventory.csv`
- `phase11g_0_kaggle_yaml_template.yaml`
- `phase11g_0_kaggle_notebook_commands.sh`
- `phase11g_0_guardrail_checks.csv`
- `phase11g_0_non_execution_manifest.json`
- `README.md`
