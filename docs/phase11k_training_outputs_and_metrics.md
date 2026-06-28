# Phase 11K Training Outputs And Metrics

Phase 11K is a strict non-execution collection phase after a completed Phase 11J.1 Kaggle training run. Its purpose is to inspect the existing local training output directory, parse `results.csv`, hash the existing weight files, and materialize small provenance artifacts only.

Phase 11K must not:

- run training
- run evaluation or validation as a separate execution step
- run inference
- mutate the original dataset
- mutate the staging dataset
- upload anything to Kaggle
- create, modify, copy, or move model weights

Inputs:

- Phase 11J.1 summary: `artifacts/phase11j1_locked_kaggle_training_execution/phase11j1_training_execution_summary.json`
- local training output directory: `phase11j_training/yolov8n_drill_bit_phase11j`

Behavior:

- If the Phase 11J.1 summary exists and shows the executed success state, Phase 11K records that the summary was available and validated.
- If the Phase 11J.1 summary is missing or still shows a local dry-run or other non-success state, Phase 11K does not fail silently. It marks `phase11j1_summary_available = false`, records the observed summary status, and falls back to direct local inspection of the provided training output directory.
- The script validates the presence of `results.csv`, `weights/best.pt`, and `weights/last.pt`.
- If `args.yaml` or related YOLO run config files are present, their paths are recorded.
- Plot and result image files are listed by path only.
- `results.csv` is parsed without assuming one fixed Ultralytics schema beyond trimmed column matching and a conservative metric-priority order for best-epoch selection:
  - `metrics/mAP50-95(B)`
  - `metrics/mAP50(B)`
  - `fitness`
- Weight manifests record path, existence, size, modified time, and streamed `sha256` hashes only.
- The output tree artifact lists relative file paths and sizes without copying binary contents.

Outputs under `artifacts/phase11k_training_outputs_and_metrics/`:

- `phase11k_training_outputs_summary.json`
- `phase11k_results_metrics_summary.csv`
- `phase11k_weight_files_manifest.csv`
- `phase11k_training_output_tree.txt`
- `phase11k_non_execution_manifest.json`
- `README.md`

Expected complete state:

- `status = phase11k_training_outputs_and_metrics_collected`
- `training_executed = false`
- `evaluation_executed = false`
- `inference_executed = false`
- `dataset_mutated = false`
- `kaggle_upload_executed = false`
- `weights_created_by_phase11k = false`
- `weights_copied_by_phase11k = false`
- `next_allowed_step = phase11l_evaluate_trained_model_on_approved_test_split`

Verification:

```bash
python -m compileall src scripts
python scripts/collect_phase11k_training_outputs_and_metrics.py \
  --phase11j1-summary artifacts/phase11j1_locked_kaggle_training_execution/phase11j1_training_execution_summary.json \
  --training-output-dir phase11j_training/yolov8n_drill_bit_phase11j
```

Git handling:

- do not add `phase11j_training/` to Git
- do not add any `.pt` file to Git
- only the small Phase 11K metadata artifacts should be staged if they need forced tracking
