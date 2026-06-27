#!/usr/bin/env bash
# Phase 11G.0 Kaggle preflight only. Do not run training from this script.

set -eu

cd /kaggle/working
git clone https://github.com/REPLACE_WITH_OWNER/XAI-small_object_detection.git
cd XAI-small_object_detection
# If a newer commit exists that includes Phase 11G.0, replace the hash below with that commit.
git checkout 1fbac0d7cf18e96a6e3ce852139305b1ffc577b9
# Example for a newer Phase 11G.0 commit:
# git checkout REPLACE_WITH_PHASE11G_0_COMMIT

export REPO_ROOT=/kaggle/working/XAI-small_object_detection
export DATA_YAML=/kaggle/working/staging_dataset_drill_bit_yolo_kaggle.yaml
export PROJECT_DIR=/kaggle/working/experiments/phase11g_staging_training_chipped

cat > "$DATA_YAML" <<'YAML'
path: /kaggle/input/REPLACE_WITH_STAGING_DATASET_SLUG/staging_dataset_copy
train: images/train
val: images/valid
test: images/test
nc: 5
names:
  0: Broken
  1: Chipped
  2: Scratched
  3: Severe_Rust
  4: Tip_Wear
YAML

test -f "$DATA_YAML"
cat "$DATA_YAML"
find /kaggle/input -maxdepth 3 -type d | sort | head -100
python - <<'PY'
from pathlib import Path
import json
import os

try:
    import yaml
except ImportError as exc:
    raise SystemExit(f'PyYAML is required for preflight: {exc}')

data_yaml = Path(os.environ['DATA_YAML'])
payload = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
root = Path(payload['path'])
report = {'data_yaml': str(data_yaml), 'path': str(root), 'splits': {}}
for split_key, split_name in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
    rel = payload[split_key]
    image_dir = root / rel
    label_dir = root / 'labels' / split_name
    images = sorted(p for p in image_dir.iterdir()) if image_dir.is_dir() else []
    labels = sorted(p for p in label_dir.iterdir()) if label_dir.is_dir() else []
    report['splits'][split_name] = {
        'image_dir_exists': image_dir.is_dir(),
        'label_dir_exists': label_dir.is_dir(),
        'image_count': len([p for p in images if p.is_file() or p.is_symlink()]),
        'label_count': len([p for p in labels if p.is_file() or p.is_symlink()]),
    }
try:
    import torch
except ImportError:
    report['torch_cuda_available'] = 'torch_not_installed'
else:
    report['torch_cuda_available'] = bool(torch.cuda.is_available())
print(json.dumps(report, indent=2))
PY

# Phase 11G training command is intentionally not executed in this preflight script.
# Example only; do not uncomment until Kaggle preflight has passed:
# yolo detect train data="$DATA_YAML" model=yolov8n.pt epochs=100 imgsz=640 batch=16 seed=42 project="$PROJECT_DIR" name=yolov8n_staging_chipped_phase11g
