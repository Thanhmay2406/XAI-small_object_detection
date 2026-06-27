# Phase 11F.1 Kaggle Input Requirements

Phase 11F.1 prepares a Kaggle execution adapter only. No training, evaluation, or inference is run here.

## Required confirmations before Phase 11G on Kaggle
- Confirm `REPO_ROOT` points to the extracted or working-copy repo root. Default template: `/kaggle/working/XAI-small_object_detection`.
- Confirm `DATA_YAML` resolves to the staging dataset YAML equivalent on Kaggle. Default template: `$REPO_ROOT/artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml`.
- Confirm `PROJECT_DIR` is writable under `/kaggle/working`. Default template: `/kaggle/working/experiments/phase11g_staging_training_chipped`.
- Confirm the staging dataset YAML remains staging-only and does not redirect to the original dataset.
- Confirm no command writes into `/kaggle/input`.
- Confirm the approved training semantics remain unchanged: `model=yolov8n.pt`, `epochs=100`, `imgsz=640`, `batch=16`, `seed=42`.

## Guardrails
- Original dataset mutation remains forbidden.
- Staging dataset mutation remains forbidden in Phase 11F.1.
- Phase 11F summary is read-only input and is not modified by this phase.
- If Kaggle repo root or dataset mount differs, override `REPO_ROOT`, `DATA_YAML`, or `PROJECT_DIR` explicitly before Phase 11G instead of editing the prepared semantics.
