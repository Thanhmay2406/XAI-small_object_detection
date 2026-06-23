# 01 — Problem Statement

## 1. Bối cảnh

Phát hiện vật thể nhỏ là một nhánh khó của object detection. Một vật thể nhỏ thường chỉ chiếm rất ít pixel trong ảnh, đặc biệt trong ảnh UAV, ảnh viễn thám, ảnh giao thông, ảnh y tế hoặc ảnh công nghiệp. Khi số pixel đại diện cho vật thể quá ít, detector dễ gặp các vấn đề:

- Không đủ đặc trưng hình dạng.
- Dễ bị nhầm với nhiễu nền.
- Dễ bị mất sau downsampling.
- Bbox lệch nhẹ cũng gây ảnh hưởng lớn đến IoU.
- Feature ở tầng sâu có semantic mạnh nhưng spatial detail yếu.

## 2. Vấn đề kỹ thuật

Một detector CNN/FPN thường đi qua nhiều bước giảm kích thước:

```text
Input image
  ↓
Backbone stage 1
  ↓
Backbone stage 2
  ↓
Backbone stage 3
  ↓
Backbone stage 4
  ↓
FPN/PAN
  ↓
Detection head
```

Với object lớn, việc giảm độ phân giải vẫn còn đủ signal. Với object nhỏ, một bbox chỉ vài pixel có thể trở thành một hoặc không có cell rõ ràng ở tầng sâu.

Ví dụ trực giác:

```text
Object 12×12 pixels ở ảnh gốc
Stride 4  → khoảng 3×3 cells
Stride 8  → khoảng 1.5×1.5 cells
Stride 16 → dưới 1 cell rõ ràng
Stride 32 → gần như mất thông tin định vị
```

## 3. Vấn đề khoa học

Các kiến trúc hiện tại thường giải quyết bằng:

- Thêm P2 head.
- Multi-scale feature fusion.
- Super-resolution branch.
- Attention module.
- Transformer query design.
- Label assignment phù hợp tiny object.

Nhưng các phương pháp này chưa trực tiếp trả lời câu hỏi:

> Trong quá trình forward, evidence của từng small object bị mất ở tầng nào, giảm bao nhiêu, và có thể regularize quá trình đó không?

## 4. Vai trò của XAI

XAI thường được dùng sau khi model đã train xong:

```text
Prediction → Explanation → Human interpretation
```

Đề tài này đổi hướng:

```text
Prediction/Feature → Explanation → Evidence metric → Training signal
```

Tức là XAI được dùng để tạo một cầu nối giữa:

- vùng object thật,
- feature map của mô hình,
- saliency/evidence của dự đoán,
- loss huấn luyện.

## 5. Định nghĩa bài toán

Cho dataset object detection gồm ảnh `x` và nhãn bbox/class `y`. Với mỗi object nhỏ `o`, ta có bbox `b_o`. Detector sinh feature maps ở nhiều tầng `F_l`, ví dụ P2/P3/P4. Một phương pháp XAI hoặc saliency function `A` sinh saliency map:

```text
S_l = A(F_l, prediction, target)
```

Ta muốn đo evidence:

```text
E_l(o) = Evidence(S_l, b_o)
```

Và giảm evidence drop:

```text
D_l(o) = max(0, E_l(o) - E_{l+1}(o) - margin)
```

Mục tiêu:

```text
L_total = L_det + λ L_evidence
```

Trong đó `L_evidence` làm giảm drop bất thường của small-object evidence qua feature levels.

## 6. Phạm vi ban đầu

Ở phiên bản đầu, chỉ tập trung vào:

- One-stage detector có FPN/PAN/P2.
- Dataset YOLO format hoặc COCO format.
- XAI dạng activation-based saliency trước, ví dụ EigenCAM-like.
- Evidence metric dựa trên saliency energy inside bbox.
- Small object theo ngưỡng diện tích bbox.

## 7. Output mong muốn

Repo cần tạo được:

1. Baseline detector report.
2. Evidence analysis report.
3. Evidence drop visualization.
4. Evidence-guided training result.
5. Ablation study.
6. Draft paper notes.
