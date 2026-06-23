# 03 — Methodology

## 1. Tổng quan phương pháp

Pipeline nghiên cứu gồm hai nhánh:

```text
Main detection branch:
Image → Detector → Predictions → Detection Loss

Evidence branch:
Feature maps → Saliency maps → Evidence metrics → Evidence Loss
```

Ở pha phân tích, evidence branch chỉ dùng để đo. Ở pha huấn luyện, evidence branch tạo thêm loss.

## 2. Baseline detector

Repo nên thiết kế abstraction để có thể dùng nhiều detector, nhưng stage đầu nên chọn một baseline đơn giản và ổn định.

Ưu tiên:

```text
YOLOv8/YOLOv11 style detector có P2/P3/P4
```

Lý do:

- dễ train,
- dễ lấy feature maps,
- phù hợp small object,
- có nhiều baseline để so sánh,
- output evaluation rõ ràng.

Không nên bắt đầu bằng DETR nếu repo mới, vì transformer detector có training pipeline và convergence phức tạp hơn.

## 3. Feature levels

Các level ban đầu:

```text
P2: high-resolution, stride 4
P3: stride 8
P4: stride 16
P5: stride 32, optional
```

Với small objects, P2/P3 là quan trọng nhất.

## 4. Saliency generation

### 4.1. Pha đầu: activation-based saliency

Dùng saliency đơn giản:

```text
S_l = normalize(mean(abs(F_l), dim=channel))
```

hoặc EigenCAM-like:

```text
S_l = first principal component of feature activations
```

Ưu điểm:

- nhanh,
- không cần backward,
- dễ debug,
- dùng được trong analysis trước.

Nhược điểm:

- chưa target-specific,
- chưa chắc phản ánh prediction cụ thể.

### 4.2. Pha sau: prediction-aware saliency

Sau khi pipeline ổn, có thể thêm:

- Grad-CAM,
- LayerCAM,
- ODAM-like method,
- detector-specific saliency.

Nhưng không nên làm ngay từ đầu.

## 5. Evidence metric

### 5.1. Inside evidence

Với saliency map `S_l` và bbox `b`, inside energy:

```text
E_inside_l(b) = sum(S_l[p] for p inside b)
```

### 5.2. Outside evidence

```text
E_outside_l(b) = sum(S_l[p] for p outside b)
```

### 5.3. Evidence ratio

```text
Evidence_l(b) = E_inside_l(b) / (E_inside_l(b) + E_outside_l(b) + eps)
```

### 5.4. Soft bbox mask

Vì small object rất nhỏ, bbox cứng có thể quá ít pixel trên feature map. Do đó nên hỗ trợ soft mask:

```text
Mask = bbox core + Gaussian/context ring
```

Ban đầu có thể dùng bbox mở rộng:

```text
expanded_bbox = bbox enlarged by scale factor 1.5 or 2.0
```

Sau đó ablation:

```text
hard bbox vs expanded bbox vs gaussian mask
```

## 6. Evidence drop

Với các level `l_i` và `l_{i+1}`:

```text
Drop(l_i → l_{i+1}) = max(0, Evidence(l_i) - Evidence(l_{i+1}))
```

Phiên bản có margin:

```text
Drop_loss(l_i → l_{i+1}) = ReLU(Evidence(l_i) - Evidence(l_{i+1}) - allowed_drop)
```

Ý nghĩa:

- Cho phép evidence giảm nhẹ.
- Phạt nếu evidence giảm quá nhanh.

## 7. Evidence Preservation Loss

Loss cơ bản:

```text
L_evidence = mean_over_small_objects_and_levels(
    ReLU(Evidence_low - Evidence_high - allowed_drop)
)
```

Total loss:

```text
L_total = L_det + λ * L_evidence
```

Trong đó:

- `L_det`: loss gốc của detector.
- `λ`: trọng số evidence loss.
- `allowed_drop`: mức giảm evidence được phép.

## 8. Các biến thể loss

### 8.1. Drop-only loss

Chỉ phạt drop:

```text
L_drop = ReLU(E_l - E_{l+1} - margin)
```

### 8.2. Inbox alignment loss

Khuyến khích saliency nằm trong bbox:

```text
L_align = 1 - Evidence_l(b)
```

### 8.3. Outside suppression loss

Phạt saliency ngoài bbox quá cao:

```text
L_outside = E_outside / (E_inside + E_outside + eps)
```

### 8.4. Cross-scale consistency

Ép saliency giữa các scale tương đồng sau resize:

```text
L_cons = distance(resize(S_P2), resize(S_P3))
```

Không làm tất cả cùng lúc. Roadmap:

```text
1. Drop-only
2. Drop + align
3. Drop + align + consistency
```

## 9. Khi nào áp dụng loss

Không nhất thiết áp dụng trên mọi object.

Có thể áp dụng cho:

- only small objects,
- hard small objects,
- false-negative-prone samples,
- top-k objects theo drop cao,
- subset batch để giảm compute.

Khởi đầu nên dùng:

```text
only small objects + P2/P3/P4 + activation saliency
```

## 10. Expected logs

Training log cần có:

```text
train/detection_loss
train/evidence_loss
train/evidence_p2
train/evidence_p3
train/evidence_p4
train/drop_p2_p3
train/drop_p3_p4
val/AP_S
val/AR_S
val/false_negative_small
```

## 11. Visualization

Cần lưu visualization cho:

- image gốc + bbox,
- saliency P2/P3/P4,
- evidence ratio từng tầng,
- evidence drop curve,
- case baseline fail vs ours success,
- case ours fail.

## 12. Phân tích kết quả

Một kết quả tốt không chỉ là:

```text
AP_S tăng
```

mà phải cho thấy:

```text
Evidence drop giảm
Saliency tập trung hơn vào object
False negative giảm
Training ổn định hơn
```
