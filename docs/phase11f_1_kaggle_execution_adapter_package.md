# Phase 11F.1 Kaggle Execution Adapter Package

Phase 11F.1 is allowed only after Phase 11F has already prepared the approved staging training execution package. This phase reads the Phase 11F summary and prepared shell command, confirms that the execution gate is ready, and produces a Kaggle-adapted command package without changing the approved training semantics.

Phase 11F.1 is prepare-only. It does not run training, evaluation, or inference, does not mutate the original dataset, does not mutate the staging dataset, and does not modify the Phase 11F summary.

The main purpose of Phase 11F.1 is to remove or parameterize local absolute paths from the Phase 11F command so the same approved semantics can be executed later on Kaggle. The command remains equivalent in these required arguments:

- `model=yolov8n.pt`
- `epochs=100`
- `imgsz=640`
- `batch=16`
- `seed=42`

The Kaggle adapter package uses parameterized variables such as `REPO_ROOT`, `DATA_YAML`, and `PROJECT_DIR` when the exact Kaggle mount layout cannot be assumed with certainty. This keeps the command honest and reusable without claiming a specific Kaggle path that has not been confirmed.

Original dataset mutation remains forbidden. Any later Phase 11G execution must still use the staging dataset YAML only, must keep all writes under writable Kaggle working paths, and must not write into `/kaggle/input`.

The next step from this package is `phase11g_execute_approved_staging_training_on_kaggle`.
