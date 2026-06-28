#!/usr/bin/env bash
set -euo pipefail

echo "This command file is a prepare-only artifact from Phase 11M.0."
echo "Do not execute it directly unless a later approved execution phase explicitly allows it."
exit 1

# Locked command below for review only:
# yolo detect val model=/home/thanhmay/workspace/XAI-small_object_detection/phase11j_training/yolov8n_drill_bit_phase11j/weights/best.pt data=/kaggle/input/datasets/thanhmay2406/phase11-staging-dataset-relabel-patch-chipped/staging_dataset_copy/staging_dataset_drill_bit_yolo.yaml split=test imgsz=640 batch=16 project=/home/thanhmay/workspace/XAI-small_object_detection/phase11m_test_eval name=yolov8n_drill_bit_phase11m_test_eval save_json=True save_conf=True plots=True
