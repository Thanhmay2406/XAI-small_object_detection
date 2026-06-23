# 00 — Research Manifest

## 1. Tên đề tài

**XAI-guided Evidence Preservation for Small Object Detection**

Tên tiếng Việt:

**Bảo toàn bằng chứng đặc trưng bằng XAI cho bài toán phát hiện vật thể nhỏ**

## 2. Tuyên bố nghiên cứu

Trong phát hiện vật thể nhỏ, các vật thể có kích thước vài pixel đến vài chục pixel dễ bị suy giảm thông tin khi đi qua downsampling, convolution sâu, feature fusion hoặc attention không phù hợp. Các kiến trúc hiện đại như FPN, PAN, BiFPN, P2 head, high-resolution backbone hoặc transformer detector cố gắng giảm vấn đề này bằng thiết kế kiến trúc. Tuy nhiên, một câu hỏi khác vẫn còn mở:

> Có thể dùng XAI như một tín hiệu huấn luyện để kiểm soát việc detector có đang giữ được “bằng chứng” của vật thể nhỏ qua các tầng feature hay không?

Đề tài này không xem XAI như công cụ hậu kiểm đơn thuần. XAI được dùng để:

1. Đo evidence của vật thể nhỏ qua feature levels.
2. Chẩn đoán evidence bị mất ở đâu.
3. Tạo training signal để giảm evidence drop.
4. Kiểm chứng xem việc bảo toàn evidence có cải thiện phát hiện vật thể nhỏ hay không.

## 3. Trọng tâm khoa học

Đề tài không nhằm tạo detector mới chỉ bằng cách thêm một block attention. Trọng tâm là một **framework học dựa trên explanation/evidence**.

Câu chuyện khoa học:

```text
Small object detection khó vì object evidence yếu và dễ biến mất.
XAI có thể đo được model evidence.
Nếu đo được evidence drop, ta có thể regularize quá trình học.
Nếu regularize đúng, detector sẽ giữ được small-object evidence tốt hơn.
Nếu giữ evidence tốt hơn, AP_S/AR_S và recall small object sẽ tăng.
```

## 4. Các khái niệm chính

### 4.1. Object evidence

Object evidence là lượng tín hiệu giải thích/saliency mà mô hình gán cho vùng chứa vật thể.

Một định nghĩa đơn giản:

```text
Evidence_l(obj) = sum(S_l inside bbox_obj) / sum(S_l over image)
```

Trong đó:

- `S_l` là saliency map tại tầng hoặc level `l`.
- `bbox_obj` là bounding box của object.
- `l` có thể là P2, P3, P4 hoặc feature map trong backbone.

### 4.2. Evidence drop

Evidence drop đo mức suy giảm evidence giữa hai tầng:

```text
Drop(P2→P3) = Evidence(P2) - Evidence(P3)
Drop(P3→P4) = Evidence(P3) - Evidence(P4)
```

Nếu drop quá lớn, có thể hiểu rằng small-object evidence đang bị xói mòn khi đi qua mạng.

### 4.3. Evidence preservation

Evidence preservation là mục tiêu giữ evidence không bị suy giảm quá nhanh qua các tầng.

Ví dụ tốt:

```text
P2: 0.82 → P3: 0.76 → P4: 0.61
```

Ví dụ xấu:

```text
P2: 0.82 → P3: 0.35 → P4: 0.07
```

## 5. Đóng góp kỳ vọng

### Contribution 1 — Diagnostic metric

Đề xuất bộ chỉ số đo small-object evidence qua các tầng feature.

### Contribution 2 — Evidence analysis protocol

Đề xuất quy trình phân tích evidence drop và liên hệ với lỗi false negative/localization error.

### Contribution 3 — Training-time regularization

Đề xuất Evidence Preservation Loss để đưa XAI vào quá trình huấn luyện.

### Contribution 4 — Empirical validation

Chứng minh rằng evidence-guided training cải thiện small-object detection trên ít nhất một hoặc nhiều dataset.

## 6. Nguyên tắc nghiên cứu

1. Heatmap không đủ để kết luận.
2. AP_S/AR_S quan trọng hơn mAP tổng.
3. Evidence metric phải được đo trước khi thêm loss.
4. Loss phải có ablation.
5. Phải chứng minh rằng cải thiện không đến từ may mắn seed hoặc augmentation ngẫu nhiên.
6. Phải có failure case analysis.
7. Phải kiểm tra chi phí tính toán.

## 7. Non-goals ở giai đoạn đầu

Không làm các việc sau ở stage đầu:

- Không thiết kế backbone mới.
- Không thêm nhiều attention block cùng lúc.
- Không thêm transformer nếu baseline chưa ổn.
- Không so sánh quá nhiều model khi chưa có diagnostic metric.
- Không dùng black-box XAI trong training loop ở pha đầu vì chi phí cao.

## 8. Câu chốt định hướng

> XAI không phải lớp trang trí sau detector; XAI là tín hiệu để kiểm soát cách detector giữ bằng chứng của vật thể nhỏ trong quá trình học.
