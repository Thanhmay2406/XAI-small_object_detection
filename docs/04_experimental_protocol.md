# 04 — Experimental Protocol

## 1. Nguyên tắc thực nghiệm

Thí nghiệm phải trả lời câu hỏi nghiên cứu, không chỉ tạo kết quả đẹp.

Mỗi experiment cần có:

```text
- Mục tiêu
- Config
- Seed
- Dataset split
- Model checkpoint
- Metrics
- Output artifacts
- Kết luận tạm thời
```

## 2. Dataset protocol

### 2.1. Dataset format

Hỗ trợ tối thiểu:

```text
YOLO format
COCO format, optional
```

YOLO format:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### 2.2. Small object definition

Cần cho phép cấu hình:

```yaml
small_object:
  mode: area_px
  area_threshold: 1024    # 32×32
  tiny_area_threshold: 256 # 16×16, optional
```

Nếu dataset resize ảnh về `imgsz`, cần tính area sau resize hoặc ghi rõ tính theo ảnh gốc.

### 2.3. Dataset inspection

Trước khi train, phải có report:

```text
- số ảnh train/val/test
- số object
- số small object
- phân bố bbox area
- phân bố class
- số object trung bình mỗi ảnh
- ảnh có nhiều small objects nhất
- class có nhiều small objects nhất
```

Script:

```text
scripts/inspect_dataset.py
```

Output:

```text
artifacts/dataset_inspection/dataset_summary.md
artifacts/dataset_inspection/bbox_area_hist.png
artifacts/dataset_inspection/class_distribution.csv
```

## 3. Baseline protocol

### 3.1. Baseline minimum

Thí nghiệm đầu tiên:

```text
E0_baseline_detector
```

Mục tiêu:

```text
Có checkpoint baseline đáng tin và evaluation đầy đủ.
```

Metrics:

```text
mAP
AP50
AP75
AP_S
AR_S
Precision
Recall
False Negative small object
```

### 3.2. Multi-seed

Tối thiểu:

```text
seed 0
```

Nếu có tài nguyên:

```text
seed 0, 1, 2
```

Không kết luận khoa học mạnh từ một seed nếu kết quả tăng rất nhỏ.

## 4. Evidence analysis protocol

### 4.1. Baseline evidence analysis

Experiment:

```text
E1_baseline_evidence_analysis
```

Input:

```text
baseline checkpoint
validation set
```

Output:

```text
baseline_evidence.csv
baseline_evidence_summary.md
saliency_visualizations/
```

CSV nên có cột:

```text
image_id
class_id
bbox_x1
bbox_y1
bbox_x2
bbox_y2
bbox_area
is_small
matched_prediction
prediction_confidence
prediction_iou
detection_status  # TP/FN/localization_error
E_P2
E_P3
E_P4
drop_P2_P3
drop_P3_P4
inside_P2
outside_P2
inside_P3
outside_P3
inside_P4
outside_P4
```

### 4.2. Correlation analysis

Cần kiểm tra:

```text
mean drop của TP vs FN
mean evidence của TP vs FN
correlation between evidence drop and IoU
correlation between evidence ratio and confidence
```

Không cần thống kê phức tạp ngay từ đầu, nhưng cần bảng rõ ràng.

## 5. Evidence loss protocol

Chỉ chạy sau khi E1 hoàn thành.

Experiment:

```text
E2_evidence_loss_basic
```

Config:

```yaml
xai_evidence:
  enabled: true
  saliency_method: activation_mean
  levels: [P2, P3, P4]
  small_only: true
  allowed_drop: 0.10
  lambda_evidence: 0.05
```

So sánh với:

```text
E0 baseline
```

Cần log:

```text
L_det
L_evidence
AP_S
AR_S
Evidence drop
```

## 6. Ablation protocol

### 6.1. Lambda ablation

```text
λ = 0.00, 0.01, 0.05, 0.10, 0.20
```

### 6.2. Level ablation

```text
P2 only
P3 only
P2+P3
P2+P3+P4
```

### 6.3. Object selection ablation

```text
small only
all objects
hard small only
```

### 6.4. Saliency method ablation

```text
activation_mean
activation_abs_mean
EigenCAM-like
GradCAM, optional
LayerCAM, optional
```

### 6.5. Mask ablation

```text
hard bbox
expanded bbox 1.5×
expanded bbox 2.0×
gaussian center mask
```

## 7. Comparison protocol

Bảng chính:

| Run | AP | AP_S | AR_S | FN Small | Drop P2→P3 | Drop P3→P4 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | | | | | | | |
| Baseline + traditional aug | | | | | | | optional |
| Baseline + evidence loss | | | | | | | main |

## 8. Visualization protocol

Cần ít nhất 4 nhóm hình:

1. Baseline TP có evidence tốt.
2. Baseline FN có evidence drop cao.
3. Ours sửa được case baseline fail.
4. Ours vẫn fail.

Mỗi hình nên gồm:

```text
Image + bbox
P2 saliency
P3 saliency
P4 saliency
Evidence values
Prediction result
```

## 9. Reporting protocol

Mỗi run phải tạo:

```text
experiments/<run_name>/config.yaml
experiments/<run_name>/metrics.json
experiments/<run_name>/summary.md
experiments/<run_name>/checkpoints/
experiments/<run_name>/visualizations/
```

Mỗi stage phải cập nhật:

```text
docs/experiment_log.md
```

## 10. Stop conditions

Dừng hoặc quay lại debug nếu:

- baseline không train ổn,
- AP_S quá thấp bất thường,
- saliency toàn zero hoặc NaN,
- evidence metric không khác nhau giữa TP/FN,
- evidence loss làm training collapse,
- kết quả tăng dưới noise nhưng không có multi-seed.
