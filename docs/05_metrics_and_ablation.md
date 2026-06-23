# 05 — Metrics and Ablation

## 1. Detection metrics

Metrics chính:

```text
mAP
AP50
AP75
AP_S
AP_M
AP_L
AR_S
Precision
Recall
```

Trong đề tài này, metric quan trọng nhất là:

```text
AP_S
AR_S
False Negative small object
```

mAP tổng có thể tăng ít hoặc không tăng mạnh. Nếu AP_S tăng rõ và FN small giảm, đó vẫn là kết quả phù hợp với mục tiêu đề tài.

## 2. Small object error metrics

### 2.1. False Negative Small

Số ground-truth small objects không được detect đúng.

```text
FN_small = count(gt_small not matched by prediction with IoU >= threshold)
```

### 2.2. Localization error small

Small object có prediction đúng class nhưng IoU thấp.

```text
localization_error = correct class but IoU < threshold
```

### 2.3. Background false positive near small object

Prediction nằm gần vùng object nhỏ nhưng không match đúng.

Metric này optional nhưng hữu ích để hiểu background confusion.

## 3. Evidence metrics

### 3.1. Inside energy

```text
inside_l = sum(S_l inside bbox)
```

### 3.2. Outside energy

```text
outside_l = sum(S_l outside bbox)
```

### 3.3. Evidence ratio

```text
E_l = inside_l / (inside_l + outside_l + eps)
```

### 3.4. Background reliance score

```text
B_l = outside_l / (inside_l + outside_l + eps)
```

Vì `B_l = 1 - E_l`, có thể chỉ cần báo cáo một trong hai.

### 3.5. Evidence drop

```text
Drop_P2_P3 = max(0, E_P2 - E_P3)
Drop_P3_P4 = max(0, E_P3 - E_P4)
```

### 3.6. Allowed drop violation

```text
Violation_P2_P3 = max(0, E_P2 - E_P3 - allowed_drop)
```

Metric này gần với loss.

## 4. Explanation quality metrics

Tối thiểu:

```text
BBox-saliency alignment
Inside/outside ratio
Pointing-game-like hit rate
```

### 4.1. Pointing hit

Nếu điểm saliency lớn nhất nằm trong bbox:

```text
hit = 1
```

Ngược lại:

```text
hit = 0
```

Với object nhỏ, có thể dùng expanded bbox để tránh quá khắt khe.

### 4.2. Saliency concentration

Tỉ lệ saliency nằm trong vùng object hoặc vùng quanh object.

```text
concentration = sum(S inside expanded bbox) / sum(S)
```

## 5. Training stability metrics

Cần log:

```text
train loss curve
val AP_S curve
evidence loss curve
grad norm, optional
NaN/Inf count
```

Kỳ vọng:

```text
Evidence loss không làm training collapse.
AP_S tăng sớm hơn hoặc ổn định hơn.
```

## 6. Compute metrics

Cần báo cáo chi phí:

```text
training time per epoch
GPU memory
inference time unchanged or not
extra time from saliency/evidence loss
```

Vì nếu XAI quá chậm, phương pháp khó thực tế.

## 7. Ablation matrix

### 7.1. Core ablation

| ID | Method | Purpose |
|---|---|---|
| A0 | Baseline | So sánh gốc |
| A1 | Baseline + saliency analysis only | Chứng minh analysis không ảnh hưởng train |
| A2 | Baseline + alignment loss | So với loss phổ biến hơn |
| A3 | Baseline + evidence drop loss | Phương pháp chính |
| A4 | Baseline + evidence drop + alignment | Kiểm tra kết hợp |

### 7.2. Lambda ablation

| λ | Expected behavior |
|---:|---|
| 0.00 | baseline |
| 0.01 | nhẹ, ít rủi ro |
| 0.05 | default đề xuất |
| 0.10 | mạnh hơn |
| 0.20 | kiểm tra over-regularization |

### 7.3. Level ablation

| Levels | Question |
|---|---|
| P2 | high-res only có đủ không? |
| P3 | mid-level có ổn không? |
| P2+P3 | drop đầu có quan trọng không? |
| P2+P3+P4 | full method |

### 7.4. Object size ablation

| Object group | Purpose |
|---|---|
| tiny only | xem tác động mạnh nhất |
| small only | mục tiêu chính |
| medium/large | kiểm tra tác dụng phụ |
| all objects | kiểm tra generalization |

### 7.5. Saliency method ablation

| Method | Speed | Faithfulness | Training-loop suitability |
|---|---|---|---|
| activation_mean | rất nhanh | thấp/vừa | tốt |
| activation_abs_mean | rất nhanh | thấp/vừa | tốt |
| EigenCAM-like | nhanh | vừa | tốt |
| GradCAM | chậm hơn | tốt hơn | cần tối ưu |
| LayerCAM | chậm hơn | tốt cho layer | cần tối ưu |

## 8. Bảng kết quả paper-ready

### Main table

| Method | AP | AP_S | AR_S | FN Small ↓ | Drop P2→P3 ↓ | Drop P3→P4 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | | | | | | |
| Ours | | | | | | |

### Ablation table

| Variant | AP_S | AR_S | Evidence Drop | Notes |
|---|---:|---:|---:|---|
| Baseline | | | | |
| Align only | | | | |
| Drop only | | | | |
| Drop + Align | | | | |

### Cost table

| Method | Train time/epoch | GPU memory | AP_S gain | Worth it? |
|---|---:|---:|---:|---|
| Baseline | | | | |
| Ours | | | | |

## 9. Kết luận metric cần đạt

Kết quả lý tưởng:

```text
AP_S tăng
AR_S tăng
FN small giảm
Evidence drop giảm
Chi phí tăng chấp nhận được
```

Kết quả vẫn có giá trị nghiên cứu nếu:

```text
Evidence drop giảm rõ nhưng AP_S tăng nhẹ
```

vì điều đó cho thấy XAI có thể điều chỉnh quá trình học, nhưng cần tối ưu thêm để chuyển thành detection gain mạnh hơn.
