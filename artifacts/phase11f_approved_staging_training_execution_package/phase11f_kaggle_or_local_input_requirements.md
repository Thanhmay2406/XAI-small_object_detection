# Phase 11F Kaggle or Local Input Requirements

Phase 11F prepares an execution package only. No training, evaluation, or inference is run here.

## Confirmed inputs
- Phase 11E real approval summary: `artifacts/phase11e_real_approval_validation/phase11e_real_approval_validation_summary.json`
- Approved command source script: `scripts/execute_phase9y_approved_staging_training.py`
- Staging dataset or YAML: `artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_copy`
- Staging dataset YAML used by the prepared command: `artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml`
- Approved training config: `configs/train/phase9v_staging_chipped_yolov8n.yaml`

## Requirements before Phase 11G
- Reconfirm that execution still targets staging-only inputs and not `data/YOLO_format`.
- Reconfirm that the prepared training command is reviewed as-is and remains non-executed until Phase 11G.
- Reconfirm there will be no original-dataset mutation, no relabel patching, and no architecture/loss edits during execution approval.
- If Kaggle is used, ensure the runtime has the staging dataset artifact, the approved config, cached or downloadable model weights, and the repo scripts needed for execution.
- If local execution is used, ensure `.venv/bin/yolo` or an equivalent approved runtime path exists before Phase 11G.

## Notes
- Training command status: Prepared from the frozen staging-training chain.
- Historical Phase 9Z.5 blocked flags are retained as context only and do not block this Phase 11F package after Phase 11E validation.
