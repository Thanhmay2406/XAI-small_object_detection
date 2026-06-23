# 02 — Research Gap and Hypotheses

## 1. Research gap

### Gap 1 — XAI trong object detection chủ yếu là hậu kiểm

Nhiều phương pháp XAI cho object detection tập trung vào việc giải thích prediction sau khi mô hình đã train xong. Chúng thường trả lời:

```text
Detector nhìn vào vùng nào?
Prediction này dựa vào feature nào?
Mô hình có nhìn vào background không?
```

Nhưng chúng ít được dùng như một phần của objective trong training detector.

### Gap 2 — Small object detection tập trung nhiều vào kiến trúc

Các nghiên cứu small object detection thường tập trung vào:

- feature pyramid,
- high-resolution representation,
- attention,
- transformer query,
- super-resolution,
- label assignment,
- tiling/cropping.

Ít công trình đặt câu hỏi:

```text
Có thể đo và bảo toàn explanation/evidence của small object qua các tầng không?
```

### Gap 3 — Thiếu metric mô tả evidence degradation

AP_S cho biết detector phát hiện tốt hay không, nhưng không cho biết:

```text
small-object evidence bị mất ở đâu?
P2 còn giữ object không?
P3 có chuyển sang background không?
P4 có còn thông tin object không?
```

Do đó cần một metric trung gian: **Evidence Drop**.

### Gap 4 — Thiếu liên kết giữa explanation quality và detection quality

Một heatmap đẹp không đảm bảo AP_S tăng. Đề tài cần kiểm chứng quan hệ:

```text
Evidence preservation ↑  →  AP_S/AR_S ↑ ?
```

## 2. Research questions

### RQ1 — Diagnostic question

> Small-object evidence có thật sự suy giảm qua P2/P3/P4 trong baseline detector không?

### RQ2 — Error relation question

> Evidence drop có liên quan đến false negative hoặc localization error của small objects không?

### RQ3 — Method question

> Evidence Preservation Loss có làm giảm evidence drop không?

### RQ4 — Performance question

> Giảm evidence drop có cải thiện AP_S, AR_S và small-object recall không?

### RQ5 — Robustness question

> Phương pháp có ổn định qua seed, dataset subset, và các ngưỡng small-object khác nhau không?

## 3. Hypotheses

### H1 — Baseline evidence degradation

Baseline detector có xu hướng mất small-object evidence ở feature levels sâu hơn.

Dự đoán:

```text
Evidence(P2) > Evidence(P3) > Evidence(P4)
```

và drop lớn hơn ở object nhỏ so với object trung bình/lớn.

### H2 — Evidence drop liên quan đến lỗi

Các object bị false negative hoặc localization error có evidence drop cao hơn các object được phát hiện đúng.

Dự đoán:

```text
Mean drop của missed small objects > Mean drop của correctly detected small objects
```

### H3 — Evidence Preservation Loss làm giảm drop

Thêm `L_evidence` giúp giữ saliency/evidence ổn định hơn qua các tầng.

Dự đoán:

```text
Drop_ours(P2→P3) < Drop_baseline(P2→P3)
Drop_ours(P3→P4) < Drop_baseline(P3→P4)
```

### H4 — Giảm drop giúp tăng AP_S/AR_S

Nếu model giữ evidence tốt hơn, recall và AP của small objects sẽ tăng.

Dự đoán:

```text
AP_S_ours > AP_S_baseline
AR_S_ours > AR_S_baseline
FalseNegativeSmall_ours < FalseNegativeSmall_baseline
```

### H5 — Evidence loss cần kiểm soát cẩn thận

Nếu `λ` quá lớn, model có thể overfit vào saliency hoặc làm giảm mAP tổng.

Dự đoán:

```text
λ nhỏ/vừa: AP_S tăng
λ quá lớn: training bất ổn hoặc precision giảm
```

## 4. Minimum publishable story

Một phiên bản tối thiểu đủ thuyết phục cần chứng minh:

1. Có hiện tượng evidence drop trong baseline.
2. Evidence drop liên quan đến lỗi small object.
3. Loss đề xuất giảm evidence drop.
4. AP_S/AR_S tăng so với baseline.
5. Ablation cho thấy cải thiện đến từ evidence preservation chứ không phải may mắn.

## 5. Failure scenarios cần chuẩn bị

### Scenario 1 — Evidence drop không rõ

Có thể do saliency method không đủ nhạy. Cách xử lý:

- thử LayerCAM/GradCAM/EigenCAM,
- hook tầng khác,
- phân tích object nhỏ hơn,
- kiểm tra saliency normalization.

### Scenario 2 — Evidence drop giảm nhưng AP_S không tăng

Có thể loss làm saliency đẹp hơn nhưng không giúp detection. Cách xử lý:

- kiểm tra correlation evidence vs detection,
- thử loss chỉ áp dụng cho hard samples,
- giảm λ,
- thêm teacher hoặc scale consistency.

### Scenario 3 — AP_S tăng nhưng evidence không cải thiện

Cải thiện có thể đến từ yếu tố khác. Cần kiểm tra:

- seed,
- augmentation,
- learning rate,
- schedule,
- confidence threshold,
- validation split.

### Scenario 4 — Chi phí quá cao

Cần có bản lightweight:

- activation-based saliency thay vì gradient-based mỗi batch,
- tính loss trên subset object,
- chỉ áp dụng loss vài epoch đầu hoặc theo interval,
- chỉ dùng P2/P3 thay vì P2/P3/P4.
