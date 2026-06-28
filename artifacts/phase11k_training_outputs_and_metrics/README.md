# Phase 11K Training Outputs And Metrics

Phase 11K is a strict non-execution provenance phase.

- status = `phase11k_training_outputs_and_metrics_collected`
- phase11j1_summary_available = `False`
- phase11j1_status = `phase11j1_execution_not_started_missing_execute_flag`
- training_output_dir = `/home/thanhmay/workspace/XAI-small_object_detection/phase11j_training/yolov8n_drill_bit_phase11j`
- results_csv_exists = `True`
- best_weight_exists = `True`
- last_weight_exists = `True`
- results_row_count = `100`
- results_csv_parsed = `True`
- best_epoch_selection_metric = `metrics/mAP50-95(B)`
- best_epoch_index = `41`
- final_epoch_index = `99`
- weights_hashed = `True`
- next_allowed_step = `phase11l_evaluate_trained_model_on_approved_test_split`

Non-execution guarantees:

- Phase 11K did not run training.
- Phase 11K did not run evaluation.
- Phase 11K did not run inference.
- Phase 11K did not mutate any dataset.
- Phase 11K did not upload anything to Kaggle.
- Phase 11K did not create or copy weights/checkpoints.

Weight files recorded by path, size, and sha256 only:

- `best.pt`: exists=`True`, size_bytes=`6257578`, sha256=`d37122f78f6a4a00a9e17ebb865b67445d2cb59c5db6ec5f6dc96a34aaa5d3e5`
- `last.pt`: exists=`True`, size_bytes=`6257578`, sha256=`6a88326fdae59ccf6ee0105f8155056b40104a8a15aca36118eef17ba4a2dc4e`

Training output tree artifact:

- truncated = `False`
- path = `/home/thanhmay/workspace/XAI-small_object_detection/artifacts/phase11k_training_outputs_and_metrics/phase11k_training_output_tree.txt`

