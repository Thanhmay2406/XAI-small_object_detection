#!/usr/bin/env bash
# This command is prepared but not executed by Phase 11F.1.
# Execute only in Phase 11G after confirming this Kaggle adapter package.
# This training must use staging dataset/config only.
# Original dataset mutation remains forbidden.

set -eu

REPO_ROOT=${REPO_ROOT:-/kaggle/working/XAI-small_object_detection}
DATA_YAML=${DATA_YAML:-$REPO_ROOT/artifacts/phase9s_staging_dataset_relabel_patch_chipped/staging_dataset_drill_bit_yolo.yaml}
PROJECT_DIR=${PROJECT_DIR:-/kaggle/working/experiments/phase11g_staging_training_chipped}

yolo detect train data="$DATA_YAML" model=yolov8n.pt epochs=100 imgsz=640 batch=16 seed=42 project="$PROJECT_DIR" name=yolov8n_staging_chipped_phase11g
