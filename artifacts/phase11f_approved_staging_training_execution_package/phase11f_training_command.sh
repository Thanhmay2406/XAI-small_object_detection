#!/usr/bin/env bash
# This command is prepared but not executed by Phase 11F.
# Execute only in Phase 11G after confirming this package.
# This training must use staging dataset/config only.
# Original dataset mutation remains forbidden.

set -eu

/home/thanhmay/workspace/XAI-small_object_detection/.venv/bin/yolo detect train data=/home/thanhmay/workspace/XAI-small_object_detection/artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 seed=42 project=experiments/phase9y_staging_training_chipped name=yolov8n_staging_chipped_phase9y
