# M8_v1c Training Entrypoint

## Scope

This document describes the repo-local M8_v1c training entrypoint that upgrades the earlier proposal into a runtime-checkable method path:

- Script: `scripts/train_m8_v1c_automated_policy_mining.py`
- Runtime module: `models/m8_v1c_runtime.py`
- Synthetic verifier: `scripts/check_m8_v1c_runtime_integration.py`
- Output dir: `artifacts/m8_v1c_training_entrypoint/`
- Runtime method config: `configs/method/m8_v1c_runtime_scale_policy.yaml`

## What Changed

`M8_v1c` is no longer only a proposal artifact.
This entrypoint adds a real repo-local runtime hook by:

1. loading an M8_v1c runtime policy config
2. subclassing the Ultralytics `DetectionTrainer`
3. subclassing `DetectionModel`
4. intercepting the feature list immediately before `Detect` consumes `P2/P3/P4/P5`
5. applying per-sample scale weights based on conservative image-level size-group inference from GT boxes

## Runtime Hook Boundary

The runtime hook is applied at the `Detect` input for the local P2 model shape:

- `Detect.from = [18, 21, 24, 27]`
- semantic mapping order: `P2, P3, P4, P5`

The hook remains repo-local and does not modify site-packages.

## Conservative Size-group Inference

Because trainer-time feature tensors are per-image while GT boxes are per-instance, the current M8_v1c runtime uses a conservative image-level reduction:

- count GT boxes per image by `small / medium / large / unknown`
- if one group is strictly dominant, use that group
- if there is a tie or no usable box, fall back to `unknown`
- `unknown` uses identity weights

This keeps the runtime path deterministic and fail-closed.

## Default Behavior

Without `--execute`, the entrypoint does not train.
It only:

- resolves dataset, model, and method-config paths
- validates dataset YAML shape
- validates the runtime policy config
- performs a synthetic forward pass through the custom model
- confirms that the runtime policy is applied before `Detect`
- writes a preflight bundle with `training_executed=false`

## Dry-run Verification Command

```bash
.venv/bin/python scripts/train_m8_v1c_automated_policy_mining.py \
  --data data/YOLO_format/data.yaml \
  --model yolov8s-p2.yaml \
  --method-config configs/method/m8_v1c_runtime_scale_policy.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device cpu \
  --workers 0 \
  --patience 20 \
  --project artifacts/m8_v1c_training_runs \
  --name m8_v1c_runtime_preflight \
  --seed 42 \
  --dry-run
```

## Execute Mode

`--execute` uses the same script and passes the custom trainer class to `YOLO.train(...)`.

At execution time, the runtime method config is injected through `method_config` and loaded by the custom trainer.

## Current Limitation

This entrypoint makes `M8_v1c` technically runnable, but it does not yet add a separate human approval-gate artifact chain like `M8_v1b.3 -> M8_v1b.4`.

So the code path is now trainable, but workflow approval is still up to the researcher before actual execution.
