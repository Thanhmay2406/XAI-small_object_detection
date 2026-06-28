# Phase 11J.1 Locked Kaggle Training Execution

Phase 11J.1 is the first approved Kaggle training execution phase for the Phase 11 path. It is allowed to execute only after Phase 11J.0 has locked the exact approved Kaggle training command.

The execution wrapper is conservative:

- without `--execute`, it stays in non-execution mode
- it validates the Phase 11J.0 summary before doing anything else
- it reads the exact locked command from `phase11j0_approved_training_command.txt`
- it requires that command to match `approved_training_command` in the Phase 11J.0 summary exactly
- it refuses to execute if the environment does not look like Kaggle

Dry-run behavior without `--execute`:

- `status = phase11j1_execution_not_started_missing_execute_flag`
- `training_executed = false`
- `evaluation_executed = false`
- `inference_executed = false`
- `weights_created = false`
- `next_allowed_step = rerun_phase11j1_with_execute_on_kaggle_runtime`

Execution behavior with `--execute`:

- the script may execute only the exact locked command
- it does not append, remove, or rewrite command arguments
- it records start time, finish time, return code, stdout log path, stderr log path, and expected output paths
- it requires `/kaggle/working` and `/kaggle/input`
- it requires the locked YAML path and locked dataset root to exist

If Kaggle runtime checks fail:

- `status = phase11j1_blocked_not_kaggle_runtime`
- `training_executed = false`

If the locked command returns `0`:

- `status = phase11j1_locked_kaggle_training_executed`
- `training_executed = true`
- `evaluation_executed = false`
- `yolo_internal_validation = true`
- `inference_executed = false`
- `dataset_mutated = false`
- `kaggle_upload_executed = false`
- `weights_created = true` only if expected YOLO weight files actually exist
- `next_allowed_step = phase11k_collect_training_outputs_and_metrics`

If the command fails:

- `status = phase11j1_locked_kaggle_training_failed`
- `training_executed = true` only if the command actually started
- stdout and stderr logs are preserved
- `weights_created` is based on actual file existence
- `next_allowed_step = inspect_phase11j1_failure_logs`

Artifacts written under `artifacts/phase11j1_locked_kaggle_training_execution/`:

- `phase11j1_training_execution_summary.json`
- `phase11j1_locked_command_used.txt`
- `phase11j1_execution_stdout.log`
- `phase11j1_execution_stderr.log`
- `phase11j1_expected_outputs_check.json`
- `phase11j1_non_mutation_manifest.json`
- `README.md`

Local verification must not run training. The expected local result is the dry-run status above. Actual Kaggle execution should be triggered manually later with `--execute`.
