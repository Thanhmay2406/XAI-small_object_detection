# Hướng dẫn triển khai repo mới — XAI-guided Evidence Preservation for Small Object Detection



---

# FILE: README.md

# XAI-guided Evidence Preservation for Small Object Detection

> Repository nghiên cứu mới cho đề tài: **Bảo toàn bằng chứng đặc trưng bằng XAI cho bài toán phát hiện vật thể nhỏ**.

## 1. Mục tiêu ngắn gọn

Đề tài này nghiên cứu việc dùng Explainable AI (XAI) không chỉ để hậu kiểm detector, mà như một **training signal** để đo, giám sát và bảo toàn “bằng chứng” của vật thể nhỏ trong quá trình trích xuất đặc trưng qua backbone/FPN.

Câu hỏi chính:

> Liệu saliency/attribution có thể trở thành tín hiệu huấn luyện giúp detector bảo toàn bằng chứng của vật thể nhỏ qua các tầng feature và cải thiện AP_S/AR_S hay không?

## 2. Tư duy nghiên cứu

Repo này không bắt đầu bằng việc tạo kiến trúc mới. Quy trình đúng là:

```text
Observation
  ↓
Scientific Question
  ↓
Hypothesis
  ↓
Diagnostic Evidence
  ↓
Method
  ↓
Experiment
  ↓
Paper-ready Evidence
```

Sai lầm cần tránh:

```text
Có ý tưởng → tạo module mới → train ngay → kết quả khó giải thích
```

Cách làm mong muốn:

```text
Có giả thuyết → đo hiện tượng → chứng minh có evidence drop → mới thiết kế loss
```

## 3. Bốn stage chính

```text
Stage 1 — Research Foundation
Stage 2 — Baseline Construction
Stage 3 — Evidence Analysis
Stage 4 — Evidence-guided Learning
```

### Stage 1 — Research Foundation

Tạo nền tảng khoa học của đề tài:

- Problem statement
- Research gap
- Research questions
- Hypotheses
- Experimental protocol
- Metric definitions
- Repo structure

### Stage 2 — Baseline Construction

Xây baseline object detector sạch, có thể lặp lại:

- Dataset loader
- YOLO/RT-DETR/Faster R-CNN wrapper hoặc Ultralytics wrapper
- Training config
- Evaluation script
- Logging
- Baseline report

Không thêm XAI/loss ở stage này.

### Stage 3 — Evidence Analysis

Dùng XAI để đo hiện tượng:

- Hook feature maps P2/P3/P4
- Sinh saliency maps
- Đo evidence trong bbox small object
- Đo evidence outside bbox
- Đo evidence drop qua tầng
- Kiểm tra correlation giữa evidence drop và lỗi detector

Không train model mới ở stage này.

### Stage 4 — Evidence-guided Learning

Sau khi chứng minh hiện tượng evidence drop, mới thêm:

- Evidence Preservation Loss
- Saliency Alignment Loss, nếu cần
- Cross-layer Evidence Consistency
- Ablation study
- So sánh với baseline và augmentation truyền thống

## 4. Cấu trúc repo đề xuất

```text
xai-evidence-small-object-detection/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── dataset/
│   ├── model/
│   ├── train/
│   └── xai_evidence/
├── docs/
│   ├── 00_research_manifest.md
│   ├── 01_problem_statement.md
│   ├── 02_research_gap_and_hypotheses.md
│   ├── 03_methodology.md
│   ├── 04_experimental_protocol.md
│   ├── 05_metrics_and_ablation.md
│   └── 06_implementation_roadmap.md
├── prompts/
│   ├── codex_master_prompt_new_repo.md
│   └── codex_phase_prompts.md
├── src/
│   └── xai_evidence_sod/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── xai/
│       ├── evidence/
│       ├── losses/
│       ├── visualization/
│       └── utils/
├── scripts/
│   ├── inspect_dataset.py
│   ├── train_baseline.py
│   ├── eval_baseline.py
│   ├── debug_feature_hooks.py
│   ├── analyze_baseline_evidence.py
│   ├── train_with_evidence_loss.py
│   └── compare_runs.py
├── experiments/
├── artifacts/
├── notebooks/
├── tests/
└── paper/
```

## 5. Kết quả kỳ vọng

Kết quả không chỉ là heatmap đẹp. Cần chứng minh bằng metric:

```text
AP_S ↑
AR_S ↑
Recall small object ↑
False Negative small object ↓
Evidence ratio inside bbox ↑
Evidence outside bbox ↓
Evidence drop P2→P3/P3→P4 ↓
Training stability ↑
```

## 6. Quy tắc làm việc

1. Không thêm loss nếu chưa có báo cáo evidence analysis.
2. Không sửa baseline khi đang thêm module nghiên cứu.
3. Mọi thí nghiệm phải có config riêng.
4. Mọi script phải có output rõ ràng.
5. Mọi kết quả phải lưu vào `experiments/` hoặc `artifacts/`.
6. Mọi kết luận khoa học phải có bảng số liệu hoặc visualization đi kèm.
7. Ưu tiên code nhỏ, kiểm thử được, hơn là code lớn nhưng khó debug.

## 7. File quan trọng nên đọc trước

```text
docs/00_research_manifest.md
docs/06_implementation_roadmap.md
prompts/codex_master_prompt_new_repo.md
prompts/codex_phase_prompts.md
```


---

# FILE: docs/00_research_manifest.md

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


---

# FILE: docs/01_problem_statement.md

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


---

# FILE: docs/02_research_gap_and_hypotheses.md

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


---

# FILE: docs/03_methodology.md

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


---

# FILE: docs/04_experimental_protocol.md

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


---

# FILE: docs/05_metrics_and_ablation.md

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


---

# FILE: docs/06_implementation_roadmap.md

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
