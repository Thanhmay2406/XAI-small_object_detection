# Phase 11M.1 Approved Test Evaluation Execution

Phase 11M.1 is the explicit execution wrapper for the locked test evaluation package prepared in Phase 11M.0.

Purpose:

1. Read the passed Phase 11M.0 package.
2. Resolve runtime model and dataset YAML paths.
3. Stay blocked by default.
4. Execute `yolo detect val` only with `--execute` or a fully approved Phase 11M.1 approval CSV.
5. Record logs, command metadata, and lightweight output metadata if execution occurs.

Why this phase follows Phase 11M.0:

- Phase 11M.0 prepared the locked command and approval artifacts without executing anything.
- Phase 11M.1 is the first phase that may actually run test evaluation, and only when explicitly authorized.

Default blocked behavior:

```bash
python scripts/execute_phase11m1_approved_test_evaluation.py
```

- verifies Phase 11M.0 metadata
- prepares the runtime command
- writes the Phase 11M.1 artifacts
- does not execute evaluation

Kaggle execution with explicit approval:

```bash
python scripts/execute_phase11m1_approved_test_evaluation.py \
  --execute \
  --model-path /kaggle/working/yolov8n_drill_bit_phase11j/weights/best.pt \
  --data-yaml /kaggle/input/datasets/thanhmay2406/phase11-staging-dataset-relabel-patch-chipped/staging_dataset_copy/staging_dataset_drill_bit_yolo.yaml \
  --project /kaggle/working/phase11m_test_eval \
  --name yolov8n_drill_bit_phase11m_test_eval
```

Runtime override behavior:

- `--model-path` lets Kaggle use the mounted or copied runtime path for `best.pt`
- `--data-yaml` lets Kaggle point at the mounted dataset YAML
- `--project`, `--name`, `--imgsz`, `--batch`, and `--split` are also overridable

Why the Kaggle dataset path is acceptable:

- Phase 11M.0 already established that the resolved dataset YAML is a Kaggle-mounted path
- this is acceptable because the intended execution target is Kaggle GPU with mounted dataset inputs
- a local non-Kaggle verification run may therefore remain blocked on dataset availability without being treated as a logic failure

What is generated if evaluation runs:

- stdout and stderr logs
- the exact command executed
- lightweight metadata about the produced evaluation output directory
- optional parsed metrics if `results.json` or `results.csv` is available

Still forbidden in Phase 11M.1:

- training
- dataset mutation
- label mutation
- weight copying
- weight modification
- direct checkpoint tensor loading in Python

Carried-forward provenance caveat:

- the repo-local Phase 11J.1 summary still reports `phase11j1_execution_not_started_missing_execute_flag`
- Phase 11K relied on direct local output inspection
- Phase 11L validated only local checkpoint-output integrity and consistency
- Phase 11M.1 carries that caveat forward and treats the checkpoint as a baseline candidate rather than a final research result

Expected next step after successful execution:

- `phase11n_collect_and_validate_test_evaluation_outputs`
