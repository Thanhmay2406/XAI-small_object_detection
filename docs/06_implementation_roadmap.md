# 06 — Implementation Roadmap for a New Repository

## Tổng quan

Roadmap này dành cho repo mới hoàn toàn. Không giả định đã có trainer, dataset loader hay model wrapper.

Triết lý:

```text
Research first, implementation second.
```

Mỗi phase có output nghiệm thu rõ ràng.

---

# Phase 0 — Initialize repository

## Mục tiêu

Tạo nền repo sạch.

## Tasks

```text
- Tạo cấu trúc thư mục.
- Tạo README.md.
- Tạo pyproject.toml hoặc requirements.txt.
- Tạo docs/.
- Tạo prompts/.
- Tạo src/xai_evidence_sod/.
- Tạo scripts/.
- Tạo tests/.
- Tạo experiments/ và artifacts/ với .gitkeep.
```

## Files expected

```text
README.md
requirements.txt
pyproject.toml, optional
docs/00_research_manifest.md
configs/default.yaml
src/xai_evidence_sod/__init__.py
```

## Acceptance criteria

```text
python -m compileall src
```

chạy không lỗi.

---

# Phase 1 — Research documentation

## Mục tiêu

Hoàn thiện tài liệu nền.

## Tasks

```text
- Viết problem statement.
- Viết research gap.
- Viết hypotheses.
- Viết methodology.
- Viết experimental protocol.
- Viết metrics and ablation plan.
```

## Acceptance criteria

```text
docs/ có đủ 6 file chính.
Mỗi file có mục tiêu rõ ràng.
Không có TODO quá mơ hồ.
```

---

# Phase 2 — Dataset foundation

## Mục tiêu

Đọc dataset và phân tích small-object distribution.

## Modules

```text
src/xai_evidence_sod/data/yolo_dataset.py
src/xai_evidence_sod/data/coco_dataset.py, optional
src/xai_evidence_sod/data/size_analysis.py
```

## Scripts

```text
scripts/inspect_dataset.py
```

## Required features

```text
- Đọc data.yaml YOLO.
- Đếm số ảnh/labels.
- Parse bbox.
- Tính area bbox theo pixel.
- Phân nhóm tiny/small/medium/large.
- Lưu histogram và CSV.
```

## Acceptance criteria

Script chạy được:

```bash
python scripts/inspect_dataset.py \
  --data data/SkyFusion_yolo/data.yaml \
  --imgsz 640 \
  --out artifacts/dataset_inspection
```

Output:

```text
artifacts/dataset_inspection/dataset_summary.md
artifacts/dataset_inspection/bbox_stats.csv
```

---

# Phase 3 — Baseline detector wrapper

## Mục tiêu

Tạo baseline train/eval sạch.

## Recommended start

Dùng Ultralytics YOLO wrapper trước để tránh viết detector từ đầu.

## Modules

```text
src/xai_evidence_sod/models/yolo_wrapper.py
src/xai_evidence_sod/evaluation/detection_metrics.py
src/xai_evidence_sod/utils/config.py
src/xai_evidence_sod/utils/seed.py
```

## Scripts

```text
scripts/train_baseline.py
scripts/eval_baseline.py
```

## Acceptance criteria

Có thể chạy:

```bash
python scripts/train_baseline.py \
  --config configs/train/baseline_yolo.yaml
```

và tạo:

```text
experiments/baseline_yolo_seed0/
├── config.yaml
├── metrics.json
├── summary.md
└── weights/
```

---

# Phase 4 — Feature hook system

## Mục tiêu

Lấy feature maps từ detector mà không làm thay đổi output.

## Modules

```text
src/xai_evidence_sod/xai/feature_hooks.py
src/xai_evidence_sod/xai/layer_registry.py
```

## Scripts

```text
scripts/inspect_model_layers.py
scripts/debug_feature_hooks.py
```

## Required features

```text
- In danh sách layer.
- Cho phép map tên P2/P3/P4 sang layer index/name.
- Register forward hook.
- Capture feature maps.
- Clear hooks sau khi dùng.
- Kiểm tra NaN/Inf.
```

## Acceptance criteria

```bash
python scripts/debug_feature_hooks.py \
  --weights experiments/baseline_yolo_seed0/weights/best.pt \
  --data data/SkyFusion_yolo/data.yaml \
  --levels P2 P3 P4
```

Output:

```text
P2 shape: [B, C, H, W]
P3 shape: [B, C, H, W]
P4 shape: [B, C, H, W]
No NaN/Inf found
```

---

# Phase 5 — Saliency maps

## Mục tiêu

Sinh saliency maps từ feature maps.

## Modules

```text
src/xai_evidence_sod/xai/saliency.py
```

## Methods v1

```text
activation_mean
activation_abs_mean
eigen_cam_like
```

## Required API

```python
saliency = compute_saliency(feature_map, method="activation_abs_mean")
```

Input:

```text
[B, C, H, W]
```

Output:

```text
[B, 1, H, W], normalized [0, 1]
```

## Script

```text
scripts/debug_saliency_one_batch.py
```

## Acceptance criteria

Output:

```text
artifacts/xai_debug/saliency_p2.png
artifacts/xai_debug/saliency_p3.png
artifacts/xai_debug/saliency_p4.png
```

---

# Phase 6 — Evidence metrics

## Mục tiêu

Đo evidence inside/outside bbox và evidence drop.

## Modules

```text
src/xai_evidence_sod/evidence/metrics.py
src/xai_evidence_sod/evidence/masks.py
```

## Required functions

```python
compute_inside_energy(saliency, bbox)
compute_outside_energy(saliency, bbox)
compute_evidence_ratio(saliency, bbox)
compute_evidence_drop(evidence_by_level)
make_bbox_mask(shape, bbox, mode="hard")
scale_bbox_to_feature(bbox, image_shape, feature_shape)
```

## Script

```text
scripts/debug_evidence_metrics.py
```

## Acceptance criteria

Tensor giả và bbox giả chạy đúng.

---

# Phase 7 — Baseline evidence analysis

## Mục tiêu

Chứng minh hoặc bác bỏ evidence drop ở baseline.

## Script

```text
scripts/analyze_baseline_evidence.py
```

## Output

```text
artifacts/evidence_analysis/baseline_evidence.csv
artifacts/evidence_analysis/baseline_evidence_summary.md
artifacts/evidence_analysis/visualizations/
```

## Acceptance criteria

Summary có:

```text
- số small objects phân tích
- mean E_P2/E_P3/E_P4
- mean drop P2→P3
- mean drop P3→P4
- TP vs FN comparison
- top failure cases
```

Không được sang Phase 8 nếu chưa có file này.

---

# Phase 8 — Evidence Preservation Loss

## Mục tiêu

Tạo loss bảo toàn evidence.

## Modules

```text
src/xai_evidence_sod/losses/evidence_loss.py
```

## Loss v1

```python
L = mean(ReLU(E_low - E_high - allowed_drop))
```

## Requirements

```text
- differentiable nếu dùng trong training.
- bật/tắt bằng config.
- lambda_evidence = 0 phải giống baseline.
- log evidence loss riêng.
```

## Acceptance criteria

Unit test:

```text
Nếu E_low=0.8, E_high=0.75, margin=0.1 → loss=0
Nếu E_low=0.8, E_high=0.3, margin=0.1 → loss>0
```

---

# Phase 9 — Training with evidence loss

## Mục tiêu

Train baseline + evidence loss.

## Scripts

```text
scripts/train_with_evidence_loss.py
```

## Config

```yaml
xai_evidence:
  enabled: true
  saliency_method: activation_abs_mean
  levels: [P2, P3, P4]
  small_only: true
  lambda_evidence: 0.05
  allowed_drop: 0.10
  mask_mode: expanded_bbox
  bbox_expand_ratio: 1.5
```

## Acceptance criteria

Run tạo:

```text
experiments/evidence_loss_seed0/
├── config.yaml
├── metrics.json
├── summary.md
├── train_curves.csv
└── weights/
```

---

# Phase 10 — Compare and report

## Mục tiêu

So sánh baseline và ours.

## Script

```text
scripts/compare_runs.py
```

## Output

```text
artifacts/comparisons/main_comparison.md
artifacts/comparisons/main_table.csv
artifacts/comparisons/evidence_drop_plot.png
```

## Acceptance criteria

Bảng có:

```text
AP
AP_S
AR_S
FN_small
E_P2
E_P3
E_P4
Drop_P2_P3
Drop_P3_P4
```

---

# Phase 11 — Ablation study

## Mục tiêu

Chứng minh method không ăn may.

## Ablations

```text
lambda
levels
mask mode
saliency method
small-only vs all objects
```

## Output

```text
artifacts/ablation/ablation_summary.md
```

---

# Phase 12 — Paper preparation

## Mục tiêu

Chuyển kết quả thành paper notes.

## Files

```text
paper/outline.md
paper/related_work.md
paper/method.md
paper/experiments.md
paper/tables/
paper/figures/
```

## Acceptance criteria

Có outline gồm:

```text
1. Introduction
2. Related Work
3. Method
4. Experiments
5. Results
6. Limitations
7. Conclusion
```
